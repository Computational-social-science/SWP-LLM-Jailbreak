"""qwen3-0.6b smoke tests — the second real model, and the first one big
enough to feel. same gating as the smollm2 file: marked slow, skipped unless
the weights are already on disk.

    .venv/bin/python -c "from huggingface_hub import snapshot_download; \
        snapshot_download('Qwen/Qwen3-0.6B')"

honestly expected an mps fight here (f16, gqa, 28 layers) and got none — the
whole capture path is architecture-agnostic as long as the model is
llama-shaped, and qwen3 is.
"""

import pytest
import torch

from logitscope.capture import CaptureConfig
from logitscope.generate import SamplingParams, generate_stream, raw_logits
from logitscope.model import load_model

from test_real_model import weights_on_disk

pytestmark = pytest.mark.slow

QWEN3 = "Qwen/Qwen3-0.6B"

needs_weights = pytest.mark.skipif(
    not weights_on_disk(QWEN3), reason="qwen3 weights not on disk"
)

PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def qwen3():
    return load_model(QWEN3)


@needs_weights
def test_generates_sensible_text(qwen3):
    ids = qwen3.tokenizer.encode(PROMPT)
    steps = list(generate_stream(qwen3, ids, SamplingParams(temperature=0, max_new_tokens=8)))
    text = qwen3.tokenizer.decode([s.token_id for s in steps])
    assert "Paris" in text


@needs_weights
def test_seeded_sampling_deterministic_on_device(qwen3):
    ids = qwen3.tokenizer.encode(PROMPT)
    p = SamplingParams(temperature=0.8, top_p=0.95, seed=42, max_new_tokens=10)
    a = [s.token_id for s in generate_stream(qwen3, ids, p)]
    b = [s.token_id for s in generate_stream(qwen3, ids, p)]
    assert a == b


@needs_weights
def test_lens_final_row_matches_real_logits(qwen3):
    """qwen3 ties its embeddings to the unembedding — a good extra check that
    the lens reads lm_head and not some stale weight reference."""
    ids = qwen3.tokenizer.encode(PROMPT)
    s = next(iter(generate_stream(
        qwen3, ids, SamplingParams(temperature=0, max_new_tokens=1),
        capture=CaptureConfig(lens=True),
    )))
    assert len(s.capture.lens) == qwen3.model.config.num_hidden_layers + 1
    lens_ids, lens_probs = s.capture.lens[-1]
    assert lens_ids[0] == s.token_id
    full = torch.softmax(raw_logits(qwen3, ids), dim=-1)
    assert lens_probs[0] == pytest.approx(float(full.max()), rel=0.05)


@needs_weights
def test_attention_capture_expands_gqa_heads(qwen3):
    """qwen3-0.6b runs gqa (16 query heads over 8 kv heads). eager attention
    materializes weights per QUERY head, so the capture should be 16 rows —
    the ui's head picker counts query heads and this keeps them aligned."""
    cfg = qwen3.model.config
    assert cfg.num_key_value_heads < cfg.num_attention_heads  # else this test is vacuous
    ids = qwen3.tokenizer.encode(PROMPT)
    s = next(iter(generate_stream(
        qwen3, ids, SamplingParams(temperature=0, max_new_tokens=1),
        capture=CaptureConfig(attn_layers=(0,)),
    )))
    heads, q, kv = s.capture.forward.attn[0].shape
    assert heads == cfg.num_attention_heads
    assert kv == len(ids)
