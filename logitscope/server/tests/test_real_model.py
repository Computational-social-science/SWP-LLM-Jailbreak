"""smoke tests against real weights on the real device (mps on my machine).

marked slow, and skipped entirely when the weights aren't already on disk —
the suite must never trigger a 270mb download on somebody else's clone.
grab the weights once with:

    .venv/bin/python -c "from huggingface_hub import snapshot_download; \
        snapshot_download('HuggingFaceTB/SmolLM2-135M-Instruct')"
"""

import pytest
import torch

from logitscope.capture import CaptureConfig
from logitscope.generate import SamplingParams, generate_stream, raw_logits
from logitscope.model import load_model

pytestmark = pytest.mark.slow

SMOLLM2 = "HuggingFaceTB/SmolLM2-135M-Instruct"


def weights_on_disk(model_id: str) -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_id, local_files_only=True)
        return True
    except Exception:
        return False


needs_weights = pytest.mark.skipif(
    not weights_on_disk(SMOLLM2), reason="smollm2 weights not on disk"
)

PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def smollm2():
    return load_model(SMOLLM2)


@needs_weights
def test_generates_sensible_text(smollm2):
    ids = smollm2.tokenizer.encode(PROMPT)
    steps = list(generate_stream(smollm2, ids, SamplingParams(temperature=0, max_new_tokens=12)))
    text = smollm2.tokenizer.decode([s.token_id for s in steps])
    assert "Paris" in text  # a 135m model that can't do this one is broken


@needs_weights
def test_greedy_deterministic_on_device(smollm2):
    ids = smollm2.tokenizer.encode(PROMPT)
    p = SamplingParams(temperature=0, max_new_tokens=10)
    a = [s.token_id for s in generate_stream(smollm2, ids, p)]
    b = [s.token_id for s in generate_stream(smollm2, ids, p)]
    assert a == b


@needs_weights
def test_seeded_sampling_deterministic_on_device(smollm2):
    """this holds on mps only because token picks happen on a cpu generator —
    mps rng is not seed-stable across runs the way cpu is."""
    ids = smollm2.tokenizer.encode(PROMPT)
    p = SamplingParams(temperature=0.8, top_p=0.95, seed=42, max_new_tokens=10)
    a = [s.token_id for s in generate_stream(smollm2, ids, p)]
    b = [s.token_id for s in generate_stream(smollm2, ids, p)]
    assert a == b


@needs_weights
def test_lens_final_row_matches_real_logits(smollm2):
    """same invariant as the tiny fixture, but with f16 slop: hidden states
    detach to f32 on cpu and requantize to f16 going back through the lens, so
    probs get a loose tolerance and only the top-1 id is required to agree."""
    ids = smollm2.tokenizer.encode(PROMPT)
    s = next(iter(generate_stream(
        smollm2, ids, SamplingParams(temperature=0, max_new_tokens=1),
        capture=CaptureConfig(lens=True),
    )))
    assert len(s.capture.lens) == smollm2.model.config.num_hidden_layers + 1
    lens_ids, lens_probs = s.capture.lens[-1]
    assert lens_ids[0] == s.token_id
    # true full-vocab probability, not softmax-over-topk — the top-32 only holds
    # ~93% of the mass here and normalizing over it alone inflates the top-1
    full = torch.softmax(raw_logits(smollm2, ids), dim=-1)
    assert lens_probs[0] == pytest.approx(float(full.max()), rel=0.05)


@needs_weights
def test_attention_capture_real_model(smollm2):
    ids = smollm2.tokenizer.encode(PROMPT)
    cfg = CaptureConfig(attn_layers=(0, 15, 29), attn_heads=(0, 4))
    steps = list(generate_stream(
        smollm2, ids, SamplingParams(temperature=0, max_new_tokens=3), capture=cfg))
    for i, s in enumerate(steps):
        attn = s.capture.forward.attn
        assert sorted(attn) == [0, 15, 29]
        for w in attn.values():
            heads, q, kv = w.shape
            assert heads == 2
            assert kv == len(ids) + i
    # eos sanity while we have a real tokenizer in hand
    assert smollm2.tokenizer.eos_token_id is not None
