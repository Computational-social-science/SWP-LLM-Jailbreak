"""measure wire bytes per token for the standard generation capture.

the question the binary protocol has to answer: how many bytes does one
generated token cost on the websocket, and how much of that is the encoding's
fault? this generates real tokens on a real model with the same capture config
the ui uses, then prices every frame four ways:

    json      the strawman: tensors as json arrays (what a lazy v0 would send)
    f32       binary, floats as f32, ids as u32, same header
    f16       binary as actually sent, before gzip
    wire      what encode_frame really produces (f16 + gzip)

run from server/ (downloads the model on first use):

    .venv/bin/python scripts/bench_payload.py [--model smollm2-135m] [--tokens 32]

writes docs/bench/payload.json and prints the table.
"""

import argparse
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from logitscope.capture import CaptureConfig
from logitscope.engine import capture_at, extend
from logitscope.generate import SamplingParams
from logitscope.model import load_model
from logitscope.service import KNOWN_MODELS
from logitscope.session import ROOT, Session, new_session_id
from logitscope.wire import decode_frame, frames_for_step

PROMPT = "The oscilloscope on the bench flickered once, and then the trace began to speak."


def frame_sizes(frame: bytes) -> dict:
    """price one encoded frame under each encoding. header is 6 bytes + 4 per dim."""
    kind, arr = decode_frame(frame)
    header = 6 + 4 * arr.ndim
    f16 = header + arr.nbytes
    f32 = header + (arr.nbytes * 2 if arr.dtype == np.float16 else arr.nbytes)
    as_json = len(json.dumps(arr.tolist(), separators=(",", ":")))
    return {"json": as_json, "f32": f32, "f16": f16, "wire": len(frame)}


def main() -> None:
    ap = argparse.ArgumentParser("bench_payload")
    ap.add_argument("--model", default="smollm2-135m", choices=sorted(KNOWN_MODELS))
    ap.add_argument("--tokens", type=int, default=32)
    args = ap.parse_args()

    bundle = load_model(KNOWN_MODELS[args.model])
    sess = Session(new_session_id(), args.model, PROMPT, bundle.tokenizer.encode(PROMPT))
    cfg = CaptureConfig(topk_logits=32, lens=True, resid_norms=True)
    params = SamplingParams(temperature=0, max_new_tokens=args.tokens)

    by_kind: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    control_bytes = 0
    n = 0
    last_node = ROOT
    for node, step in extend(bundle, sess, ROOT, params, capture=cfg):
        meta, frames = frames_for_step(step.capture)
        # the json control message rides along with every token's frames
        control_bytes += len(json.dumps({"type": "token", "node": node.to_dict(),
                                         "frames": meta}, separators=(",", ":")))
        for desc, frame in zip(meta, frames):
            sizes = frame_sizes(frame)
            for enc, b in sizes.items():
                by_kind[desc["kind"]][enc] += b
        last_node = node.id
        n += 1

    kinds = {k: {enc: b / n for enc, b in v.items()} for k, v in by_kind.items()}
    totals = {enc: sum(v[enc] for v in kinds.values()) for enc in ("json", "f32", "f16", "wire")}

    # one attention scrub frame, priced separately: it's per click, not per token,
    # and it dwarfs everything else
    mid = bundle.model.config.num_hidden_layers // 2
    sc = capture_at(bundle, sess, last_node, CaptureConfig(topk_logits=8, attn_layers=(mid,)))
    meta, frames = frames_for_step(sc)
    attn_i = next(i for i, d in enumerate(meta) if d["kind"] == "attn")
    _, attn_arr = decode_frame(frames[attn_i])
    # a node's capture covers the context that predicted it, i.e. up to its parent
    ctx = int(attn_arr.shape[-1])
    attn = {"shape": list(attn_arr.shape), "ctx": ctx, **frame_sizes(frames[attn_i])}

    out = {
        "model": KNOWN_MODELS[args.model],
        "device": str(bundle.device),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "prompt_tokens": len(sess.prompt_ids),
        "gen_tokens": n,
        "capture": {"topk_logits": 32, "lens": True, "resid_norms": True},
        "per_token_by_kind": kinds,
        "per_token_total": totals,
        "control_msg_bytes_per_token": control_bytes / n,
        "attn_scrub_frame": attn,
    }
    dest = Path(__file__).resolve().parents[2] / "docs" / "bench" / "payload.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")

    print(f"{args.model} on {bundle.device}, {n} tokens, capture topk32+lens+resid\n")
    print(f"{'frame kind':<14}{'json':>10}{'f32':>10}{'f16':>10}{'wire':>10}")
    for k, v in kinds.items():
        print(f"{k:<14}" + "".join(f"{v[e]:>10.0f}" for e in ("json", "f32", "f16", "wire")))
    print(f"{'total B/tok':<14}" + "".join(
        f"{totals[e]:>10.0f}" for e in ("json", "f32", "f16", "wire")))
    print(f"\ncontrol msg: {control_bytes / n:.0f} B/tok")
    print(f"attn scrub frame {attn['shape']} @ ctx {ctx}: "
          f"json {attn['json']}, f32 {attn['f32']}, f16 {attn['f16']}, wire {attn['wire']}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
