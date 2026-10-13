"""scrub-back latency: activation lru hit vs a full recompute forward.

scrubbing history in the ui sends a `capture` message per hover/click. if the
lru is warm the server answers from memory; cold means one whole forward over
the node's prefix with hooks attached. this measures both, through the real
websocket endpoint (TestClient, so the asgi stack is included), on the real
model. run from server/:

    .venv/bin/python scripts/bench_scrub.py

writes docs/bench/cache.json and prints the summary.
"""

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from logitscope.app import create_app
from logitscope.cache import ActivationLRU
from logitscope.model import load_model
from logitscope.service import KNOWN_MODELS, Service
from logitscope.session import ROOT

PROMPT = "The oscilloscope on the bench flickered once, and then the trace began to speak."
GEN_CAPTURE = {"topk_logits": 32, "lens": True, "resid_norms": True}
TOKENS = 32
SCRUB_NODES = 16


def drain_reply(ws) -> dict:
    """read one capture reply (json + its binary frames), return the json."""
    msg = json.loads(ws.receive()["text"])
    assert msg["type"] == "capture", msg
    for _ in msg["frames"]:
        ws.receive()
    return msg


def timed_scrub(ws, sid: str, node: int, capture: dict) -> tuple[float, bool]:
    t0 = time.perf_counter()
    ws.send_json({"type": "capture", "session_id": sid, "node_id": node,
                  "capture": capture})
    msg = drain_reply(ws)
    return (time.perf_counter() - t0) * 1000, msg["cached"]


def scrub_all(ws, sid: str, nodes: list[int], capture: dict, expect_cached: bool) -> list[float]:
    times = []
    for node in nodes:
        ms, cached = timed_scrub(ws, sid, node, capture)
        assert cached == expect_cached, f"node {node}: cached={cached}, expected {expect_cached}"
        times.append(ms)
    return times


def summary(times: list[float]) -> dict:
    return {"median_ms": statistics.median(times),
            "mean_ms": statistics.fmean(times),
            "n": len(times)}


def main() -> None:
    name = "smollm2-135m"
    bundle = load_model(KNOWN_MODELS[name])
    svc = Service()
    svc.register_bundle(name, bundle)
    client = TestClient(create_app(svc))

    sess = client.post("/api/sessions", json={"model": name, "prompt": PROMPT}).json()
    sid = sess["id"]
    mid = bundle.model.config.num_hidden_layers // 2
    attn_capture = {"topk_logits": 8, "attn_layers": [mid]}

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "generate", "session_id": sid, "from_node": ROOT,
                      "params": {"temperature": 0, "max_new_tokens": TOKENS},
                      "capture": GEN_CAPTURE})
        nodes = []
        while True:
            msg = json.loads(ws.receive()["text"])
            if msg["type"] == "done":
                break
            for _ in msg["frames"]:
                ws.receive()
            nodes.append(msg["node"]["id"])
        nodes = nodes[-SCRUB_NODES:]  # the deep ones, where recompute hurts most

        # replay-path scrubs first: generation just warmed exactly these keys.
        # (order matters — wiping the lru later kills them.)
        gen_hit = scrub_all(ws, sid, nodes, GEN_CAPTURE, expect_cached=True)

        # attention scrubs were never captured during generation, so the first
        # pass is all recomputes and the second is all hits. the warmup uses a
        # different layer so its cache key doesn't collide with the bench's.
        timed_scrub(ws, sid, nodes[0], {"topk_logits": 8, "attn_layers": [0]})
        attn_miss = scrub_all(ws, sid, nodes, attn_capture, expect_cached=False)
        attn_hit = scrub_all(ws, sid, nodes, attn_capture, expect_cached=True)

        # now wipe the lru and force the generation capture to recompute
        svc.cache = ActivationLRU(svc.cache.max_bytes)
        gen_miss = scrub_all(ws, sid, nodes, GEN_CAPTURE, expect_cached=False)

    ctx = len(sess["prompt_ids"]) + TOKENS
    out = {
        "model": KNOWN_MODELS[name],
        "device": str(bundle.device),
        "context_len": ctx,
        "scrubbed_nodes": len(nodes),
        "attn_scrub": {"capture": attn_capture,
                       "recompute": summary(attn_miss), "hit": summary(attn_hit)},
        "gen_scrub": {"capture": GEN_CAPTURE,
                      "recompute": summary(gen_miss), "hit": summary(gen_hit)},
    }
    dest = Path(__file__).resolve().parents[2] / "docs" / "bench" / "cache.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")

    for label, d in (("attn scrub", out["attn_scrub"]), ("gen scrub", out["gen_scrub"])):
        r, h = d["recompute"]["median_ms"], d["hit"]["median_ms"]
        print(f"{label}: recompute {r:.1f} ms, lru hit {h:.2f} ms  ({r / h:.0f}x)")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
