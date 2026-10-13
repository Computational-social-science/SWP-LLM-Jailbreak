import pytest
import torch

from logitscope.capture import CaptureConfig
from logitscope.generate import SamplingParams, generate_stream

N_LAYERS = 2


def lens_steps(tiny, n=3, **cfg):
    ids = tiny.tokenizer.encode("through the stack")
    p = SamplingParams(temperature=0, max_new_tokens=n)
    return list(generate_stream(tiny, ids, p, capture=CaptureConfig(lens=True, **cfg)))


def test_lens_shape(tiny):
    for s in lens_steps(tiny):
        rows = s.capture.lens
        assert len(rows) == N_LAYERS + 1  # embeddings + each layer
        for ids, probs in rows:
            assert len(ids) == 8
            assert len(probs) == 8
            assert probs == sorted(probs, reverse=True)
            assert all(0.0 <= p <= 1.0 for p in probs)


def test_lens_topk_configurable(tiny):
    for s in lens_steps(tiny, lens_topk=3):
        assert all(len(ids) == 3 for ids, _ in s.capture.lens)


def test_lens_final_layer_equals_actual_logits(tiny):
    """the sanity anchor: the lens at the last layer goes through the exact same
    norm + unembed the model itself uses, so it must reproduce the real
    distribution. this is the test that caught the missing final norm."""
    prompt = tiny.tokenizer.encode("through the stack")
    s = next(iter(generate_stream(
        tiny, prompt, SamplingParams(temperature=0, max_new_tokens=1),
        capture=CaptureConfig(lens=True, lens_topk=8),
    )))
    with torch.no_grad():
        logits = tiny.model(input_ids=torch.tensor([prompt])).logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    want_vals, want_ids = torch.topk(probs, 8)

    got_ids, got_vals = s.capture.lens[-1]
    assert got_ids == [int(i) for i in want_ids]
    assert got_vals == pytest.approx([float(v) for v in want_vals], rel=1e-4)


def test_lens_does_not_leak_hidden_unless_asked(tiny):
    for s in lens_steps(tiny):
        assert s.capture.forward.hidden is None
    for s in lens_steps(tiny, hidden=True):
        assert s.capture.forward.hidden is not None
