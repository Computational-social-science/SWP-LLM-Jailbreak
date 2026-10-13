import pytest

from logitscope.engine import extend, fork
from logitscope.generate import SamplingParams, generate_stream
from logitscope.session import ROOT, Session, new_session_id


def fresh(tiny, prompt="fork around and find out"):
    return Session(new_session_id(), tiny.model_id, prompt, tiny.tokenizer.encode(prompt))


GREEDY = SamplingParams(temperature=0, max_new_tokens=6)


def test_extend_matches_plain_stream(tiny):
    s = fresh(tiny)
    nodes = [n for n, _ in extend(tiny, s, ROOT, GREEDY)]
    direct = [r.token_id for r in generate_stream(tiny, s.prompt_ids, GREEDY)]
    assert [n.token_id for n in nodes] == direct
    assert s.path_ids(nodes[-1].id) == direct


def test_extend_records_stats(tiny):
    s = fresh(tiny)
    for node, step in extend(tiny, s, ROOT, GREEDY):
        assert node.logprob == step.logprob
        assert node.rank == 1  # greedy
        assert node.text != ""


def test_fork_prefix_matches_parent(tiny):
    s = fresh(tiny)
    trunk = [n for n, _ in extend(tiny, s, ROOT, GREEDY)]
    at = trunk[3]  # fork the 4th token
    alt = 42 if trunk[3].token_id != 42 else 43
    branch = [n for n, _ in fork(tiny, s, at.id, alt, GREEDY)]

    # same prefix up to the fork point, then the forced token
    assert s.path_ids(branch[0].id)[:-1] == s.path_ids(at.id)[:-1]
    assert branch[0].token_id == alt
    # sibling: same parent as the token it replaces
    assert branch[0].parent == at.parent
    assert set(s.nodes[at.parent].children) >= {at.id, branch[0].id}


def test_fork_leaves_original_branch_alone(tiny):
    s = fresh(tiny)
    trunk = [n for n, _ in extend(tiny, s, ROOT, GREEDY)]
    before = s.path_ids(trunk[-1].id)
    fork_gen = fork(tiny, s, trunk[2].id, 7, GREEDY)
    list(fork_gen)
    assert s.path_ids(trunk[-1].id) == before


def test_fork_forced_token_gets_honest_stats(tiny):
    s = fresh(tiny)
    trunk = [n for n, _ in extend(tiny, s, ROOT, GREEDY)]
    # force the token the model itself would have picked -> rank must be 1
    branch = list(fork(tiny, s, trunk[0].id, trunk[0].token_id, GREEDY))
    forced = branch[0][0]
    assert forced.rank == 1
    assert forced.logprob == pytest.approx(trunk[0].logprob, rel=1e-5)


def test_fork_respects_max_new_tokens(tiny):
    s = fresh(tiny)
    trunk = [n for n, _ in extend(tiny, s, ROOT, GREEDY)]
    branch = list(fork(tiny, s, trunk[0].id, 9, GREEDY))
    assert len(branch) <= GREEDY.max_new_tokens


def test_fork_root_rejected(tiny):
    s = fresh(tiny)
    with pytest.raises(ValueError):
        list(fork(tiny, s, ROOT, 5, GREEDY))


def test_fork_bad_token_rejected(tiny):
    s = fresh(tiny)
    trunk = [n for n, _ in extend(tiny, s, ROOT, GREEDY)]
    with pytest.raises(ValueError):
        list(fork(tiny, s, trunk[0].id, 9999, GREEDY))


def test_params_fork_by_extending_again(tiny):
    """extending the same node twice = two branches, deterministic per seed."""
    s = fresh(tiny)
    a = [n.token_id for n, _ in extend(tiny, s, ROOT, SamplingParams(seed=1, max_new_tokens=5))]
    b = [n.token_id for n, _ in extend(tiny, s, ROOT, SamplingParams(seed=2, max_new_tokens=5))]
    assert len(s.nodes[ROOT].children) == 2
    assert a != b
