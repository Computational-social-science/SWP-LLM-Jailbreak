import numpy as np
import pytest
from fastapi.testclient import TestClient

from logitscope.app import create_app
from logitscope.service import Service
from logitscope.session import ROOT
from logitscope.wire import KIND_TOPK_IDS, decode_frame


@pytest.fixture()
def rig(tiny):
    svc = Service()
    svc.register_bundle("tiny-random", tiny)
    return TestClient(create_app(svc)), svc


def drain(ws):
    """read token messages + their binary frames until done."""
    tokens = []
    while True:
        msg = ws.receive_json()
        if msg["type"] == "done":
            return tokens, msg
        assert msg["type"] == "token"
        frames = [decode_frame(ws.receive_bytes()) for _ in msg["frames"]]
        tokens.append((msg["node"], msg["frames"], frames))


def test_generate_streams_tokens_and_frames(rig):
    client, svc = rig
    sess = svc.create_session("tiny-random", "stream me")
    with client.websocket_connect("/ws") as ws:
        ws.send_json({
            "type": "generate", "session_id": sess.id,
            "params": {"temperature": 0, "max_new_tokens": 4},
            "capture": {"lens": True, "resid_norms": True},
        })
        tokens, done = drain(ws)

    assert done["tokens"] == 4
    assert len(tokens) == 4
    node, meta, frames = tokens[0]
    kinds = [m["kind"] for m in meta]
    assert kinds == ["topk_ids", "topk_logits", "lens_ids", "lens_probs", "resid_norms"]
    wire_kind, topk_ids = frames[0]
    assert wire_kind == KIND_TOPK_IDS
    # greedy: first topk id is the chosen token
    assert int(topk_ids[0]) == node["token_id"]
    # the session tree actually grew
    assert len(sess.nodes) == 5


def test_generate_without_capture_sends_no_frames(rig):
    client, svc = rig
    sess = svc.create_session("tiny-random", "bare")
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "generate", "session_id": sess.id,
                      "params": {"temperature": 0, "max_new_tokens": 3}})
        tokens, done = drain(ws)
    assert all(meta == [] for _, meta, _ in tokens)


def test_fork_over_ws(rig):
    client, svc = rig
    sess = svc.create_session("tiny-random", "branch me")
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "generate", "session_id": sess.id,
                      "params": {"temperature": 0, "max_new_tokens": 4}})
        trunk, _ = drain(ws)
        at = trunk[1][0]  # second generated node
        alt = (at["token_id"] + 1) % 256
        ws.send_json({"type": "fork", "session_id": sess.id,
                      "node_id": at["id"], "token_id": alt,
                      "params": {"temperature": 0, "max_new_tokens": 3}})
        branch, done = drain(ws)

    assert done["tokens"] == 3
    forced = branch[0][0]
    assert forced["token_id"] == alt
    assert forced["parent"] == at["parent"]  # sibling of the replaced node


def test_attention_frames_carry_layer(rig):
    client, svc = rig
    sess = svc.create_session("tiny-random", "look here")
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "generate", "session_id": sess.id,
                      "params": {"temperature": 0, "max_new_tokens": 2},
                      "capture": {"attn_layers": [0, 1]}})
        tokens, _ = drain(ws)
    _, meta, frames = tokens[0]
    attn_meta = [m for m in meta if m["kind"] == "attn"]
    assert [m["layer"] for m in attn_meta] == [0, 1]
    for _, arr in frames[2:]:
        assert arr.dtype == np.float16


def test_bad_session_reports_error(rig):
    client, _ = rig
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "generate", "session_id": "nope"})
        msg = ws.receive_json()
    assert msg["type"] == "error"
    assert "nope" in msg["error"]


def test_unknown_type_reports_error(rig):
    client, _ = rig
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "dance"})
        msg = ws.receive_json()
    assert msg["type"] == "error"


def test_socket_survives_an_error(rig):
    """an error frame shouldn't kill the connection — the ui retries in place."""
    client, svc = rig
    sess = svc.create_session("tiny-random", "still here")
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "generate", "session_id": "ghost"})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "generate", "session_id": sess.id,
                      "params": {"temperature": 0, "max_new_tokens": 1}})
        tokens, done = drain(ws)
    assert done["tokens"] == 1
