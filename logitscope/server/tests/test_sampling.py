import math

import pytest
import torch

from logitscope.sampling import (
    NEG_INF,
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    greedy_token,
    sample_token,
)


def survivors(logits):
    return {i for i, v in enumerate(logits.tolist()) if v != NEG_INF}


# --- temperature ---


def test_temperature_scales():
    l = torch.tensor([2.0, 4.0])
    assert torch.allclose(apply_temperature(l, 2.0), torch.tensor([1.0, 2.0]))


def test_temperature_zero_rejected():
    with pytest.raises(ValueError):
        apply_temperature(torch.tensor([1.0]), 0.0)


# --- top-k ---


def test_top_k_keeps_k():
    l = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
    assert survivors(apply_top_k(l, 2)) == {1, 4}


def test_top_k_off_when_zero_or_big():
    l = torch.tensor([1.0, 2.0, 3.0])
    assert survivors(apply_top_k(l, 0)) == {0, 1, 2}
    assert survivors(apply_top_k(l, 99)) == {0, 1, 2}


def test_top_k_ties_all_kept():
    l = torch.tensor([3.0, 3.0, 3.0, 1.0])
    # kth value is 3.0 and all three tie for it
    assert survivors(apply_top_k(l, 2)) == {0, 1, 2}


# --- top-p, the cursed one ---


def test_top_p_one_keeps_everything():
    l = torch.tensor([1.0, 2.0, 3.0])
    assert survivors(apply_top_p(l, 1.0)) == {0, 1, 2}


def test_top_p_zero_is_greedy():
    l = torch.tensor([1.0, 2.0, 3.0])
    assert survivors(apply_top_p(l, 0.0)) == {2}


def test_top_p_smallest_covering_set():
    # probs 0.5, 0.25, 0.125, 0.125 by construction
    p = torch.tensor([0.5, 0.25, 0.125, 0.125])
    l = torch.log(p)
    assert survivors(apply_top_p(l, 0.5)) == {0}
    assert survivors(apply_top_p(l, 0.6)) == {0, 1}
    assert survivors(apply_top_p(l, 0.75)) == {0, 1}
    assert survivors(apply_top_p(l, 0.76)) == {0, 1, 2}


def test_top_p_single_token_mass():
    # one token hogs ~all the mass -> nucleus is just that token
    l = torch.tensor([100.0, 0.0, 0.0, 0.0])
    assert survivors(apply_top_p(l, 0.9)) == {0}


def test_top_p_uniform_ties():
    # 4 equal tokens, p=0.5 -> exactly 2 survive (0.25 + 0.25 covers it)
    l = torch.zeros(4)
    assert len(survivors(apply_top_p(l, 0.5))) == 2


def test_top_p_masked_mass_renormalizes():
    l = torch.tensor([2.0, 1.0, 0.0, -1.0])
    out = apply_top_p(l, 0.7)
    probs = torch.softmax(out, dim=-1)
    assert math.isclose(float(probs.sum()), 1.0, rel_tol=1e-6)
    assert float(probs[torch.softmax(l, -1).argmax()]) > 0.5


# --- repetition penalty ---


def test_rep_penalty_positive_logit_divided():
    l = torch.tensor([2.0, 4.0])
    out = apply_repetition_penalty(l, [1], 2.0)
    assert out[1].item() == pytest.approx(2.0)
    assert out[0].item() == pytest.approx(2.0)  # untouched


def test_rep_penalty_negative_logit_multiplied():
    l = torch.tensor([-2.0, 4.0])
    out = apply_repetition_penalty(l, [0], 2.0)
    assert out[0].item() == pytest.approx(-4.0)


def test_rep_penalty_noop_cases():
    l = torch.tensor([1.0, 2.0])
    assert torch.equal(apply_repetition_penalty(l, [], 2.0), l)
    assert torch.equal(apply_repetition_penalty(l, [0, 1], 1.0), l)


def test_rep_penalty_duplicate_ids_applied_once():
    l = torch.tensor([4.0, 1.0])
    out = apply_repetition_penalty(l, [0, 0, 0], 2.0)
    assert out[0].item() == pytest.approx(2.0)


# --- picking ---


def test_greedy_is_argmax():
    assert greedy_token(torch.tensor([0.1, 9.0, 3.0])) == 1


def test_sample_respects_mask():
    l = torch.tensor([0.0, NEG_INF, 0.0, NEG_INF])
    g = torch.Generator().manual_seed(123)
    picks = {sample_token(l, g) for _ in range(50)}
    assert picks <= {0, 2}


def test_sample_seed_reproducible():
    l = torch.randn(64)
    a = [sample_token(l, torch.Generator().manual_seed(9)) for _ in range(1)]
    b = [sample_token(l, torch.Generator().manual_seed(9)) for _ in range(1)]
    assert a == b
