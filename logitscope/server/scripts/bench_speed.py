"""tok/s benchmark: both models, mps vs cpu, prefill vs decode, capture cost.

three questions, one script:
  - how fast is a single prefill forward over a long prompt? (256 tokens, one
    forward, median of 3 after a warmup)
  - how fast is steady-state decode? (64 greedy tokens off a short prompt,
    timed from the first token so the prefill doesn't pollute the number)
  - what does capture cost? decode again with the ui's generation capture
    (topk32 + lens + resid norms), then with an attention layer on top.

dtype follows model.py policy: f16 on mps, f32 on cpu — that's what the
server actually runs, so that's what gets measured. run from server/:

    .venv/bin/python scripts/bench_speed.py

writes docs/bench/speed.json and prints the tables. takes a few minutes,
mostly qwen on cpu.
"""

import gc
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from logitscope.capture import CaptureConfig
from logitscope.generate import SamplingParams, generate_stream, raw_logits
from logitscope.model import load_model
from logitscope.service import KNOWN_MODELS

PASSAGE = (
    "The storm had been building over the ridge all afternoon, and by the time "
    "the first drops hit the tin roof of the observatory the wind was already "
    "pulling at the shutters. Inside, the old telescope turned slowly on its "
    "mount, tracking a point of light that had not been in last night's charts. "
) * 8
PREFILL_LEN = 256
DECODE_PROMPT_LEN = 16
DECODE_TOKENS = 64


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def bench_prefill(bundle, ids: list[int]) -> float:
    raw_logits(bundle, ids)  # warmup: mps compiles its kernels on first shape
    times = []
    for _ in range(3):
        sync(bundle.device)
        t0 = time.perf_counter()
        raw_logits(bundle, ids)
        sync(bundle.device)
        times.append(time.perf_counter() - t0)
    return len(ids) / statistics.median(times)


def bench_decode(bundle, ids: list[int], capture: CaptureConfig | None) -> float:
    """tok/s from the first generated token — the prefill pays for itself."""
    warm = SamplingParams(temperature=0, max_new_tokens=8)
    for _ in generate_stream(bundle, ids, warm, capture=capture):
        pass
    params = SamplingParams(temperature=0, max_new_tokens=DECODE_TOKENS)
    n = 0
    t0 = 0.0
    for _ in generate_stream(bundle, ids, params, capture=capture):
        if n == 0:
            t0 = time.perf_counter()
        n += 1
    elapsed = time.perf_counter() - t0
    if n < DECODE_TOKENS:
        print(f"  (eos after {n} tokens)")
    if n < 2:
        raise RuntimeError("model stopped immediately, decode number is junk")
    return (n - 1) / elapsed


def main() -> None:
    devices = [torch.device("mps")] if torch.backends.mps.is_available() else []
    devices.append(torch.device("cpu"))
    results = {"torch": torch.__version__, "machine": platform.machine(), "runs": []}

    for name, hf_id in KNOWN_MODELS.items():
        for device in devices:
            print(f"== {name} on {device} ==")
            bundle = load_model(hf_id, device=device)
            tok = bundle.tokenizer
            prefill_ids = tok.encode(PASSAGE)[:PREFILL_LEN]
            decode_ids = tok.encode(PASSAGE)[:DECODE_PROMPT_LEN]
            n_layers = bundle.model.config.num_hidden_layers

            run = {
                "model": name,
                "device": device.type,
                "dtype": str(next(bundle.model.parameters()).dtype),
                "prefill_tokens": len(prefill_ids),
                "prefill_tok_s": bench_prefill(bundle, prefill_ids),
                "decode_tok_s": bench_decode(bundle, decode_ids, None),
            }
            if device.type == "mps":
                lens = CaptureConfig(topk_logits=32, lens=True, resid_norms=True)
                attn = CaptureConfig(topk_logits=32, lens=True, resid_norms=True,
                                     attn_layers=(n_layers // 2,))
                run["decode_lens_tok_s"] = bench_decode(bundle, decode_ids, lens)
                run["decode_lens_attn_tok_s"] = bench_decode(bundle, decode_ids, attn)
            results["runs"].append(run)
            print("  " + json.dumps(run))

            del bundle
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()

    dest = Path(__file__).resolve().parents[2] / "docs" / "bench" / "speed.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(results, indent=2) + "\n")

    print(f"\n{'model':<14}{'device':<8}{'prefill tok/s':>14}{'decode tok/s':>14}")
    for r in results["runs"]:
        print(f"{r['model']:<14}{r['device']:<8}{r['prefill_tok_s']:>14.0f}"
              f"{r['decode_tok_s']:>14.1f}")
    print(f"\n{'model':<14}{'off':>8}{'lens':>8}{'+attn':>8}   (decode tok/s, mps)")
    for r in results["runs"]:
        if "decode_lens_tok_s" in r:
            print(f"{r['model']:<14}{r['decode_tok_s']:>8.1f}"
                  f"{r['decode_lens_tok_s']:>8.1f}{r['decode_lens_attn_tok_s']:>8.1f}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
