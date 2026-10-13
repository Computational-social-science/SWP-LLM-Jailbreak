"""record a full ws session against the tiny model into a jsonl fixture.

the recording captures every byte the server sends — json messages as-is,
binary frames base64'd — grouped by the request that produced them, so a
replay server can perform the whole thing back without torch. run from
server/:

    .venv/bin/python scripts/record_replay.py [out.jsonl]

regenerating the fixture is safe: the tiny model is seeded and generation is
greedy, so the bytes only change if the protocol changes (which is exactly
when you want the fixture to change).
"""

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from logitscope.app import create_app
from logitscope.service import Service
from logitscope.session import ROOT
from logitscope.tinymodel import build_tiny_bundle
from logitscope.wire import KIND_LENS_IDS, KIND_TOPK_IDS, decode_frame

PROMPT = "the scope hums: "
GEN_CAPTURE = {"topk_logits": 32, "lens": True, "resid_norms": True}
PARAMS = {"temperature": 0, "max_new_tokens": 8}
FORK_AT = 3  # node id the e2e forks at (3rd generated token)
FIXED_ID = "replay000000"  # stands in for the random session id


def normalized(sess: dict) -> dict:
    """strip the two nondeterministic fields so rerunning the recorder writes
    byte-identical fixtures (a diff means the protocol actually changed)."""
    return {**sess, "id": FIXED_ID, "created": 0.0}


def drain(ws, group: str, out: list) -> list[dict]:
    """pull messages until done (generate/fork) or one capture reply, recording
    every wire payload verbatim."""
    nodes = []
    pending_frames = 0
    saw_reply = False
    while True:
        raw = ws.receive()
        if "text" in raw:
            out.append({"kind": "ws", "group": group, "text": raw["text"]})
            msg = json.loads(raw["text"])
            if msg["type"] == "token":
                nodes.append(msg["node"])
                pending_frames = len(msg["frames"])
            elif msg["type"] == "capture":
                pending_frames = len(msg["frames"])
                saw_reply = True
            elif msg["type"] == "done":
                return nodes
            elif msg["type"] == "error":
                raise RuntimeError(f"recorded an error: {msg['error']}")
        else:
            out.append({"kind": "ws", "group": group,
                        "frame_b64": base64.b64encode(raw["bytes"]).decode()})
            pending_frames -= 1
        if saw_reply and pending_frames == 0:
            return nodes


def ids_in_frames(out: list) -> set[int]:
    """every token id a frontend could ask the decode endpoint about."""
    ids: set[int] = set()
    for rec in out:
        if "frame_b64" not in rec:
            continue
        kind, arr = decode_frame(base64.b64decode(rec["frame_b64"]))
        if kind in (KIND_TOPK_IDS, KIND_LENS_IDS):
            ids.update(int(i) for i in arr.reshape(-1))
    return ids


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "replay.jsonl"

    bundle = build_tiny_bundle(seed=0)
    svc = Service()
    svc.register_bundle("tiny-random", bundle)
    client = TestClient(create_app(svc))

    out: list[dict] = [{"kind": "meta", "prompt": PROMPT, "params": PARAMS}]

    sess = client.post("/api/sessions", json={"model": "tiny-random", "prompt": PROMPT}).json()
    sid = sess["id"]
    out.append({"kind": "models", "data": client.get("/api/models").json()})
    out.append({"kind": "session", "data": normalized(sess)})

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "generate", "session_id": sid, "from_node": ROOT,
                      "params": PARAMS, "capture": GEN_CAPTURE})
        trunk = drain(ws, "generate", out)

        # the fork the e2e performs: second-best candidate at node FORK_AT.
        # find it in the recorded topk frame for that node.
        idx = next(i for i, n in enumerate(trunk) if n["id"] == FORK_AT)
        frames_at = [r for r in out if r.get("group") == "generate" and "frame_b64" in r]
        topk_kind, topk = decode_frame(base64.b64decode(frames_at[idx * 5]["frame_b64"]))
        assert topk_kind == KIND_TOPK_IDS
        alt = int(topk[1])
        out.append({"kind": "meta", "fork_node": FORK_AT, "fork_token": alt})
        ws.send_json({"type": "fork", "session_id": sid, "node_id": FORK_AT,
                      "token_id": alt, "params": PARAMS, "capture": GEN_CAPTURE})
        branch = drain(ws, "fork", out)

        # scrub replies: attention squares for both tips on both tiny layers
        # (the attention tab), and the plain generation capture for every node
        # (the replay button).
        tips = [trunk[-1]["id"], branch[-1]["id"]]
        for tip in tips:
            for layer in (0, 1):
                ws.send_json({"type": "capture", "session_id": sid, "node_id": tip,
                              "capture": {"topk_logits": 8, "attn_layers": [layer]}})
                drain(ws, f"capture:{tip}:{layer}", out)
        for node in trunk + branch:
            ws.send_json({"type": "capture", "session_id": sid, "node_id": node["id"],
                          "capture": GEN_CAPTURE})
            drain(ws, f"capture:{node['id']}:none", out)

    decode_ids = ids_in_frames(out)
    decode_ids.update(sess["prompt_ids"])
    decode_ids.update(n["token_id"] for n in trunk + branch)
    out.append({"kind": "decode",
                "map": {str(i): bundle.tokenizer.decode([i]) for i in sorted(decode_ids)}})
    out.append({"kind": "session_final",
                "data": normalized(client.get(f"/api/sessions/{sid}").json())})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes, "
          f"{len(trunk)} trunk + {len(branch)} branch tokens, fork at {FORK_AT} -> {alt})")


if __name__ == "__main__":
    main()
