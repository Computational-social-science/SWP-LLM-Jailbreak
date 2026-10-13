import math

import pytest
import torch

from logitscope.generate import SamplingParams, generate_stream, step_stats


def run(bundle, prompt, params):
    ids = bundle.tokenizer.encode(prompt)
    return [s.token_id for s in generate_stream(bundle, ids, params)]


def test_same_seed_same_tokens(tiny):
    p = SamplingParams(temperature=0.9, top_k=40, top_p=0.95, seed=1234, max_new_tokens=16)
    assert run(tiny, "the quick brown", p) == run(tiny, "the quick brown", p)


def test_different_seed_different_tokens(tiny):
    a = run(tiny, "the quick brown", SamplingParams(seed=1, max_new_tokens=16))
    b = run(tiny, "the quick brown", SamplingParams(seed=2, max_new_tokens=16))
    assert a != b  # 16 tokens over vocab 256, collision would be absurd luck


def test_max_new_tokens_respected(tiny):
    assert len(run(tiny, "abc", SamplingParams(max_new_tokens=5))) <= 5


def test_kv_cache_matches_full_context(tiny):
    """the oracle: greedy decode with the cache must match recomputing the whole
    context from scratch every step. if the cache is wired wrong this explodes."""
    prompt = tiny.tokenizer.encode("watch it think")
    cached = run(tiny, "watch it think", SamplingParams(temperature=0, max_new_tokens=10))

    context = list(prompt)
    for expect in cached:
        with torch.no_grad():
            logits = tiny.model(input_ids=torch.tensor([context])).logits[0, -1, :].float()
        assert int(logits.argmax()) == expect
        context.append(expect)


def test_eos_stops_generation(tiny):
    # force eos as the very first pick by making it the argmax via a fat rep
    # penalty on everything else? no clean way with a random model, so instead
    # just check the contract on a stream that does emit eos.
    p = SamplingParams(temperature=1.5, seed=0, max_new_tokens=2000)
    toks = run(tiny, "x", p)
    eos = tiny.tokenizer.eos_token_id
    if eos in toks:
        assert toks.index(eos) == len(toks) - 1  # nothing generated past eos
    else:
        assert len(toks) == 2000  # never hit eos, ran to the cap


def test_empty_prompt_rejected(tiny):
    with pytest.raises(ValueError):
        list(generate_stream(tiny, [], SamplingParams()))


def test_step_stats_uniform_entropy():
    raw = torch.zeros(256)
    logprob, rank, entropy = step_stats(raw, 17)
    assert entropy == pytest.approx(math.log(256), rel=1e-5)
    assert logprob == pytest.approx(math.log(1 / 256), rel=1e-5)


def test_step_stats_ranks():
    raw = torch.tensor([0.0, 3.0, 1.0, 2.0])
    assert step_stats(raw, 1)[1] == 1  # argmax
    assert step_stats(raw, 0)[1] == 4  # worst
    assert step_stats(raw, 3)[1] == 2


def test_step_stats_peaked_entropy_near_zero():
    raw = torch.full((64,), -100.0)
    raw[5] = 100.0
    _, rank, entropy = step_stats(raw, 5)
    assert rank == 1
    assert entropy < 1e-6


def test_greedy_stream_rank_is_always_one(tiny):
    ids = tiny.tokenizer.encode("rank check")
    for s in generate_stream(tiny, ids, SamplingParams(temperature=0, max_new_tokens=8)):
        assert s.rank == 1
        assert s.logprob <= 0.0
        assert 0.0 <= s.entropy <= math.log(256) + 1e-6


def test_rep_penalty_covers_prompt_and_generated(tiny):
    """with an absurd penalty nothing seen so far should ever win argmax again,
    so greedy output has no repeats and never echoes a prompt token."""
    prompt = tiny.tokenizer.encode("hello scope")
    toks = run(tiny, "hello scope", SamplingParams(temperature=0, repetition_penalty=1e6, max_new_tokens=12))
    assert len(set(toks)) == len(toks)
    assert not (set(toks) & set(prompt))
