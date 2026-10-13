"""the replay server against the committed fixture — what the e2e runs on."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from logitscope.replay import Recording, capture_key, create_replay_app
from logitscope.wire import KIND_ATTN, KIND_TOPK_IDS, decode_frame

FIXTURE = Path(__file__).parent / "fixtures" / "replay.jsonl"


@pytest.fixture(scope="module")
def client():
    return TestClient(create_replay_app(FIXTURE))


def drain(ws):
    tokens = []
    while True:
        msg = ws.receive_json()
        if msg["type"] in ("done", "error", "capture"):
            frames = []
            if msg["type"] == "capture":
                frames = [decode_frame(ws.receive_bytes()) for _ in msg["frames"]]
            return tokens, msg, frames
        assert msg["type"] == "token"
        frames = [decode_frame(ws.receive_bytes()) for _ in msg["frames"]]
        tokens.append((msg["node"], frames))


def test_recording_loads():
    rec = Recording(FIXTURE)
    assert rec.session["id"] == "replay000000"
    assert "generate" in rec.groups
    assert "fork" in rec.groups
    assert len(rec.decode) > 50


def test_rejects_a_non_recording(tmp_path):
    bogus = tmp_path / "nope.jsonl"
    bogus.write_text('{"kind": "models", "data": []}\n')
    with pytest.raises(ValueError, match="doesn't look like a recording"):
        Recording(bogus)


def test_rest_surface_matches_fixture(client):
    assert client.get("/api/health").json() == {"ok": True, "replay": True}
    models = {m["name"]: m for m in client.get("/api/models").json()}
    assert models["tiny-random"]["loaded"] is True
    assert models["tiny-random"]["n_layers"] == 2

    sess = client.post("/api/sessions", json={"model": "x", "prompt": "ignored"}).json()
    assert sess["id"] == "replay000000"
    assert sess["nodes"][0]["id"] == 0  # root-only at create time

    final = client.get("/api/sessions/replay000000").json()
    assert len(final["nodes"]) == 17  # root + 8 trunk + forced + 7 branch


def test_decode_endpoint_uses_recorded_map(client):
    known = client.post("/api/models/tiny-random/decode", json={"ids": [104]}).json()
    assert known["texts"] == ["h"]
    unknown = client.post("/api/models/tiny-random/decode", json={"ids": [123456]}).json()
    assert unknown["texts"] == ["?"]


def test_generate_replays_the_trunk(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "generate", "session_id": "replay000000",
                      "params": {"whatever": True}})
        tokens, done, _ = drain(ws)
    assert done == {"type": "done", "tokens": 8}
    assert len(tokens) == 8
    node, frames = tokens[0]
    assert node["id"] == 1
    assert frames[0][0] == KIND_TOPK_IDS
    # greedy recording: top-1 of the topk frame is the token that was picked
    assert int(frames[0][1][0]) == node["token_id"]


def test_fork_replays_the_branch(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "fork", "session_id": "replay000000",
                      "node_id": 3, "token_id": 205})
        tokens, done, _ = drain(ws)
    assert done["tokens"] == 8
    forced = tokens[0][0]
    assert forced["token_id"] == 205
    assert forced["parent"] == 2  # sibling of node 3


def test_capture_lookup_by_node_and_layer(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "capture", "session_id": "replay000000",
                      "node_id": 8, "capture": {"topk_logits": 8, "attn_layers": [1]}})
        _, msg, frames = drain(ws)
    assert msg["type"] == "capture"
    assert msg["node_id"] == 8
    attn = [f for f in frames if f[0] == KIND_ATTN]
    assert len(attn) == 1
    heads, q, kv = attn[0][1].shape
    assert q == kv  # scrub recomputes give the full square


def test_unrecorded_request_is_an_error_not_a_crash(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "capture", "session_id": "replay000000",
                      "node_id": 999, "capture": {"attn_layers": [0]}})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "capture:999:0" in msg["error"]
        # and the socket still works after
        ws.send_json({"type": "generate", "session_id": "replay000000"})
        tokens, done, _ = drain(ws)
        assert done["tokens"] == 8


def test_capture_key_shapes():
    assert capture_key({"node_id": 4, "capture": {"attn_layers": [1]}}) == "capture:4:1"
    assert capture_key({"node_id": 4, "capture": {"lens": True}}) == "capture:4:none"
    assert capture_key({"node_id": 4}) == "capture:4:none"
