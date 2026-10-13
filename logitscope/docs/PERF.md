# measured performance

Everything here was measured on my machine: Apple M2 Pro, 16 GB, macOS 26.1,
torch 2.12.1. Each table regenerates with one script from `server/`, and the
raw output lands in `docs/bench/*.json` (committed, so you can diff my run
against yours):

```sh
.venv/bin/python scripts/bench_speed.py     # -> bench/speed.json
.venv/bin/python scripts/bench_payload.py   # -> bench/payload.json
.venv/bin/python scripts/bench_scrub.py     # -> bench/cache.json
```

A caveat that applies to every number: this server runs transformers with
**eager attention** (sdpa never materializes the attention weights, and
capturing those weights is the whole point) and a hand-rolled python
generation loop. These are honest numbers for *this instrument*, not for an
optimized inference stack.

## tok/s: prefill vs decode, mps vs cpu

256-token prompt for prefill (single forward, median of 3), 64 greedy tokens
for decode, timed from the first token so prefill doesn't pollute it. f16 on
mps, f32 on cpu — same policy the server uses.

| model | device | prefill tok/s | decode tok/s |
|---|---|---:|---:|
| SmolLM2-135M | mps | 3842 | 6.4 |
| SmolLM2-135M | cpu | 385 | 10.1 |
| Qwen3-0.6B | mps | 1778 | 10.1 |
| Qwen3-0.6B | cpu | 154 | 4.1 |

The table that taught me how mps actually works. Prefill is compute-bound, so
mps wins by 10x. Batch-1 decode on models this small is **dispatch-latency
bound** — every layer is a parade of tiny kernel launches, and the work per
kernel is nowhere near big enough to hide them. The 135M (30 thin layers)
decodes *faster on cpu than on mps*. The 0.6B has enough work per layer for
mps to pull ahead again. For an instrument that streams tokens to your
eyeballs, 6–10 tok/s is honestly a fine viewing speed, but it was not the
table I expected to publish.

## the capture tax

Same decode measurement, mps, with the capture the UI actually requests
during generation (top-32 logits + logit lens + residual norms), then with
one attention layer on top:

| model | capture off | + lens | + lens + attn |
|---|---:|---:|---:|
| SmolLM2-135M | 6.4 | 1.8 | 1.8 |
| Qwen3-0.6B | 10.1 | 1.6 | 1.8 |

Lens capture costs 3–5x: per token it hauls 29–31 hidden states off-device
and runs an unembedding matmul per layer. This tax is why capture is
toggleable, why attention is capture-on-demand only, and why the activation
cache exists — you pay once per token, not once per hover.

## payload: what one token costs on the wire

Per generated token, standard generation capture on SmolLM2 (32 tokens
averaged). "json" is the strawman v0 (tensors as json arrays), "f32"/"f16"
are the binary framing at those dtypes, "wire" is what the server actually
sends (f16 + per-frame gzip):

| frame kind | json | f32 | f16 | wire |
|---|---:|---:|---:|---:|
| topk ids | 153 | 138 | 138 | 138 |
| topk logits | 304 | 138 | 74 | 92 |
| lens ids | 1022 | 1006 | 1006 | 330 |
| lens probs | 4433 | 1006 | 510 | 530 |
| resid norms | 225 | 134 | 72 | 93 |
| **total B/tok** | **6137** | **2422** | **1800** | **1182** |

5.2x smaller than json, 2x smaller than binary f32. The json control message
riding alongside adds ~275 B/tok. Two details that surprised me: gzip *hurts*
on the tiny already-random frames (topk logits: 74 → 92 — compression headers
aren't free), and it's the id arrays that compress beautifully (lens ids:
1006 → 330, the same candidates repeat layer after layer). Gzip pays for
itself on exactly the frames that dominate.

The real monster is the on-demand attention square: one layer, 9 heads at
context 51 is **280 KiB as json, 23 KiB on the wire** (12x). That frame is
per-click, not per-token, and it's the reason the binary path exists at all.

## scrub-back: activation cache hit vs recompute

Scrubbing history re-requests per-node captures. Cold means one full forward
over the node's prefix with hooks attached; warm is a dict lookup plus
shipping the already-encoded frames. Measured through the real websocket
endpoint (asgi stack included), SmolLM2 on mps, 16 nodes at contexts up to ~50:

| scrub | recompute (median) | lru hit (median) | speedup |
|---|---:|---:|---:|
| generation capture (lens + norms) | 467.8 ms | 1.3 ms | **354x** |
| attention square (1 layer) | 146.5 ms | 6.2 ms | 24x |

The attention hit is slower than the lens hit because the reply still carries
a 23 KiB frame. Either way: scrubbing back through history went from "watch
the spinner" to imperceptible, which is the difference between an instrument
you play with and one you wait on.
