# logitscope

> **Note (this snapshot).** This is the upstream logitscope snapshot used for the
> mechanistic probe in Appendix D of the accompanying anonymous submission. We
> extended it with the comparison panel (`web/src/compare.ts`,
> `web/src/components/CompareAttention.tsx`) and the scripts that walk its
> statistics (`web/walk.mts`, `web/walk2.mts`, `web/scan.mts`, `web/layers.mts`).
> The upstream project is <https://github.com/SiddharthaG09/logitscope> and
> declares **no license**; we claim no ownership over it. Full attribution and
> the complete list of files we changed are in `../THIRD_PARTY.md`.

An oscilloscope for transformers. Run a small open model locally, type a
prompt, and watch it think, token by token: what it almost said, how sure it
was, which layer changed its mind, where it looked, and what would have
happened if it had said something else. 100% local, no API keys — weights come
straight from Hugging Face and everything after that stays on your machine.

I built this because I wanted to actually understand what happens inside an
inference stack — KV caches, sampling filters, attention, the residual
stream — and reading about them wasn't sticking. Building the instrument
forced me to hold the whole pipeline in my head at once.

> The screenshots below refer to `docs/*.png`, which are **not** included in this
> snapshot (see `../THIRD_PARTY.md`).

![stream view: token chips colored by entropy, top-k candidates for the selected token](docs/stream.png)

![logit lens waterfall: layers up the y axis, the answer crystallizing through the stack](docs/waterfall.png)

![attention: ribbons drawn from the hovered token back onto the text, heatmap below](docs/attention.png)

![fork diff: two timelines side by side after forcing a different token](docs/forkdiff.png)

## what it does

- **Stream view** — generated text as chips, colored by the entropy of the
  distribution each one was drawn from. Click a chip and the detail panel
  shows the top-k candidates with exact probabilities: what it almost said.
- **Logit lens waterfall** — the signature view. For every generated token,
  every layer's intermediate guess, drawn as a column up the stack. Early
  layers babble; somewhere in the middle the real answer crystallizes.
- **Attention** — a per-layer heatmap over the whole context, and ribbons
  drawn from a hovered token back onto the text itself, showing where the
  model looked when it wrote that word.
- **Fork explorer** — click any token, pick one of its rejected candidates,
  and generation branches right there. Branches live in a tree; any two show
  side by side with a token-level diff. "What if the model had said X" as a
  game.
- **Compare mode** — the same instrument pointed at two inputs instead of one
  output. Two prompts get one prefill each — nothing is generated — and every
  readout the main panel has comes back paired: how each tokenizes (a
  character-level diff, whitespace visible), the next-token distribution drawn
  as two bars per candidate with the JS divergence between them, the logit lens
  per layer with the first layer the two disagree on, residual norms per layer,
  and the prompt-only attention squares with a per-cell delta heatmap. Built
  for "does one extra whitespace change what the model thinks" — the answer is
  usually measurable, and sometimes yes.
- **The verdict** — next to those squares, and *first* in the panel, compare mode
  answers the question the pairing was made to ask: **can one character perturb
  the layer globally, or does it only re-tokenize the token it landed on?** The
  panel opens with that question answered — No / Partly / Yes plus the reach it
  moved — and one quiet line saying what it means for the model; the numbers
  live in the evidence below, not in the sentence. Reach is measured, not
  listed: every key is matched through the character alignment, so a token the
  edit *split* is excluded from the comparison instead of being scored as
  attention that moved — its mass is re-tokenization, not a change of target.
  What is left is the honest signal: what share of the layer's attention mass
  changed target, how many query rows were re-drawn, and how much of the
  displaced mass landed on positions the edit never touched. Underneath, each
  bullet is graded high / mid / low and carries its number, so the claim can be
  checked against the panels beside it. The copy button puts answer, scope and
  evidence on the clipboard in that order. (Floating-point floors are
  deliberate: below ~0.5% of the layer, "where did the mass go" is f16
  quantization noise, and the panel stays quiet instead of reporting 70% of
  nothing.)
- **Sessions** — every generation is a tree where each token knows its
  parent. Save/load as JSONL, replay a session, scrub back through history.

Models: SmolLM2-135M-Instruct (fast default) and Qwen3-0.6B, on Apple MPS
with CPU fallback. Any HF causal LM id with the llama-family layout works
through the load endpoint.

## the logit lens, in four sentences

A transformer doesn't conjure its answer in the last layer — every layer
nudges an evolving guess that gradually becomes the output. The logit lens
decodes that intermediate state at each layer with the model's own final norm
and unembedding, as if the network had stopped there. Early layers produce
noise, middle layers produce plausible-but-wrong candidates, and at some
depth the actual answer snaps into place and survives to the top. The
waterfall draws exactly that, one column per token. (Practical note: apply
the final norm *before* unembedding — skip it and every layer decodes to
garbage. I know because I skipped it.)

## how it's built, and why hand-rolled

**Generation loop** (`server/logitscope/generate.py`) — explicit forward
passes with the KV cache owned by the loop, not `model.generate()`. The whole
point is seeing inside every step: capture hooks, per-step stats, forced
tokens for forking, one-token-at-a-time streaming — none of it fits through
`generate()`'s interface. Temperature/top-k/top-p/repetition penalty are
implemented and unit-tested in-repo (top-p edge cases are evil), with a
seeded generator so runs reproduce.

**Capture** (`capture.py`) — real forward hooks on the decoder layers,
attached per pass and torn down after. Attention is capture-on-demand only —
one layer × head set per request — because keeping every map is
`layers × heads × ctx²` floats per step; even the 135M at 1k context would be
over 1 GiB per step. The model runs eager attention on purpose: sdpa and
flash never materialize the weights.

**Wire protocol** (`wire.py`) — JSON control frames plus binary tensor frames
(f16, per-frame gzip, 6-byte header + shape). Tensors as JSON text cost ~13
bytes a float; the binary path takes a generated token from 6.1 KiB to
1.2 KiB and a one-layer attention square from 280 KiB to 23 KiB. Frames are
byte-deterministic (gzip mtime zeroed) — the golden decoder fixtures and
replay mode both depend on that.

**Activation LRU** (`cache.py`) — captures keyed by (session, node, capture
config), byte-budgeted, storing already-encoded frames. Scrubbing history
costs a lookup instead of a forward pass: 1.3 ms instead of 468 ms.

**Frontend** (`web/`) — Vite + React + TS; every visualization hand-rolled on
Canvas2D, crisp at device pixel ratio: chips, probability bars, entropy
sparkline, waterfall, heatmap, ribbons. No chart library, because none of
these are charts a library has. The store is `useSyncExternalStore` and
nothing else.

**Replay mode** — the server records every byte it sends into a JSONL fixture
(`--record`), then performs it back with no model loaded (`--replay`, no
torch import, sub-second startup). The Playwright e2e drives the full UI
against a committed recording: deterministic, zero downloads.

## measured

Full tables and methodology in [docs/PERF.md](docs/PERF.md); raw run output in
`docs/bench/*.json`. Headlines, on an M2 Pro (16 GB):

- **Prefill vs decode:** prefill is compute-bound, MPS wins 10× (3842 tok/s
  on the 135M). Batch-1 decode at this size is dispatch-latency bound — the
  135M decodes *faster on CPU* (10.1 vs 6.4 tok/s); the 0.6B is big enough
  for MPS to win again.
- **Capture tax:** lens capture costs 3–5× in decode speed (~31 hidden states
  hauled to CPU plus 31 unembedding matmuls per token). That's why capture is
  toggleable and cached.
- **Payload:** 6137 B/token as JSON → 1182 B on the wire (f16 + gzip), and
  gzip turns out to *hurt* on tiny high-entropy frames while paying for
  itself exactly on the big repetitive ones.
- **Scrub-back:** activation-cache hit vs recompute is 1.3 ms vs 467.8 ms
  (354×) for lens captures, 6.2 vs 146.5 ms for attention squares.

## mps, honestly

Things this project taught me the hard way:

- Small-model decode is latency-bound, not compute-bound; below some size MPS
  loses to the CPU sitting next to it. Prefill meanwhile is 10× the other way.
- f16 on MPS, f32 on CPU — half matmuls on CPU are slow and occasionally
  wrong.
- Capturing attention requires eager attention, which costs real speed. An
  inspection tool should make that trade, but it should also write it down.
- Anything a hook captures must be `.detach().cpu()`'d immediately, or it
  pins the MPS graph and memory climbs until the machine complains.

## run it

Needs python 3.12 via [uv](https://docs.astral.sh/uv/) and node 22+.

```sh
make setup        # uv sync (server) + npm install (web)
make dev          # tiny random model + vite together — instant, zero downloads
```

Open http://localhost:5173, type a prompt, hit enter. For real models:

```sh
make server       # server on :8000 (first smollm2 session downloads ~270 MB)
make web          # vite on :5173, second terminal
```

Pick `qwen3-0.6b` in the model picker and hit load for the bigger model
(~1.4 GB, progress bar included). Tests and artifacts:

```sh
make test         # server pytest + web vitest
make e2e          # playwright against the committed replay fixture — no weights
make bench        # regenerate docs/bench/*.json (real model, a few minutes)
make shots        # regenerate the four screenshots above
```

The test suite runs on an in-process 2-layer random model and downloads
nothing; the real-model smoke tests (`pytest -m slow`) only run when weights
are already in the HF cache.

## what's next

- **Activation patching** — copy a hidden state from one run into another and
  see what flips. The capture/session plumbing is most of the way there.
- **Head ablations** — zero one head's output and diff the distribution shift.
- **Multi-model diff** — same prompt through both models, waterfalls side by
  side.
- **Shareable sessions** — the JSONL export already replays; a static viewer
  page would make "look what it almost said" links possible.
