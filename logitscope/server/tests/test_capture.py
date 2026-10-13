import pytest
import torch

from logitscope.capture import CaptureConfig, Recorder, topk_snapshot
from logitscope.generate import SamplingParams, generate_stream

N_LAYERS = 2
HIDDEN = 64


def steps(tiny, cfg, n=4, prompt="capture me"):
    ids = tiny.tokenizer.encode(prompt)
    p = SamplingParams(temperature=0, max_new_tokens=n)
    return list(generate_stream(tiny, ids, p, capture=cfg))


def test_no_capture_by_default(tiny):
    ids = tiny.tokenizer.encode("plain")
    out = list(generate_stream(tiny, ids, SamplingParams(max_new_tokens=3)))
    assert all(s.capture is None for s in out)


def test_topk_shapes_and_order(tiny):
    out = steps(tiny, CaptureConfig(topk_logits=32))
    for s in out:
        c = s.capture
        assert len(c.topk_ids) == 32
        assert len(c.topk_logits) == 32
        assert c.topk_logits == sorted(c.topk_logits, reverse=True)
        # greedy: the chosen token is the top-1
        assert c.topk_ids[0] == s.token_id


def test_topk_clamps_to_vocab(tiny):
    out = steps(tiny, CaptureConfig(topk_logits=10_000), n=1)
    assert len(out[0].capture.topk_ids) == 256


def test_hidden_states_captured(tiny):
    out = steps(tiny, CaptureConfig(hidden=True))
    for s in out:
        h = s.capture.forward.hidden
        assert len(h) == N_LAYERS + 1  # embeddings + each decoder layer
        for t in h:
            assert t.shape == (HIDDEN,)
            assert t.dtype == torch.float32
            assert t.device.type == "cpu"


def test_resid_norms_captured(tiny):
    out = steps(tiny, CaptureConfig(resid_norms=True))
    for s in out:
        norms = s.capture.forward.resid_norms
        assert len(norms) == N_LAYERS + 1
        assert all(n > 0 for n in norms)
        # hidden wasn't requested, so it shouldn't be kept
        assert s.capture.forward.hidden is None


def test_hooks_removed_after_stream(tiny):
    steps(tiny, CaptureConfig(hidden=True))
    # a fresh forward must not hit any leftover hooks
    rec = Recorder(tiny.model, CaptureConfig(hidden=True))
    with torch.no_grad():
        tiny.model(input_ids=torch.tensor([[1, 2, 3]]))
    assert rec._hidden == {}


def test_capture_off_leaves_no_hooks(tiny):
    before = len(tiny.model.model.layers[0]._forward_hooks)
    steps(tiny, CaptureConfig())  # config with nothing enabled
    after = len(tiny.model.model.layers[0]._forward_hooks)
    assert before == after == 0


def test_attention_selected_layers_only(tiny):
    out = steps(tiny, CaptureConfig(attn_layers=(1,)), n=3, prompt="attend!")
    prompt_len = len("attend!")
    for i, s in enumerate(out):
        attn = s.capture.forward.attn
        assert set(attn.keys()) == {1}
        heads, q, kv = attn[1].shape
        assert heads == 4
        assert attn[1].dtype == torch.float16
        if i == 0:
            # step 0's capture comes from the prefill forward
            assert (q, kv) == (prompt_len, prompt_len)
        else:
            assert q == 1
            assert kv == prompt_len + i  # context grows one token per step


def test_attention_rows_are_distributions(tiny):
    out = steps(tiny, CaptureConfig(attn_layers=(0, 1)), n=2)
    for s in out:
        for w in s.capture.forward.attn.values():
            sums = w.float().sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-2)


def test_attention_head_slicing(tiny):
    out = steps(tiny, CaptureConfig(attn_layers=(0,), attn_heads=(1, 3)), n=2)
    for s in out:
        assert s.capture.forward.attn[0].shape[0] == 2


def test_attention_off_when_not_asked(tiny):
    out = steps(tiny, CaptureConfig(hidden=True), n=2)
    assert all(s.capture.forward.attn is None for s in out)


def test_attention_layer_out_of_range(tiny):
    with pytest.raises(ValueError):
        steps(tiny, CaptureConfig(attn_layers=(7,)), n=1)


def test_topk_snapshot_matches_torch():
    raw = torch.randn(100)
    ids, vals = topk_snapshot(raw, 5)
    tv, ti = torch.topk(raw, 5)
    assert ids == [int(i) for i in ti]
    assert vals == [float(v) for v in tv]
