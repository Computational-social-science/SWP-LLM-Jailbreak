import pytest

from logitscope.session import ROOT, Session, new_session_id


def make():
    return Session(new_session_id(), "tiny-random", "hi", [104, 105])


def test_root_exists():
    s = make()
    assert s.nodes[ROOT].parent is None
    assert s.nodes[ROOT].token_id is None


def test_append_chain_and_path():
    s = make()
    a = s.append(ROOT, 10)
    b = s.append(a.id, 11)
    c = s.append(b.id, 12)
    assert s.path_ids(c.id) == [10, 11, 12]
    assert s.context_ids(c.id) == [104, 105, 10, 11, 12]
    assert s.nodes[a.id].children == [b.id]


def test_branching_makes_siblings():
    s = make()
    a = s.append(ROOT, 10)
    b1 = s.append(a.id, 11)
    b2 = s.append(a.id, 99)  # sibling of b1
    assert s.nodes[a.id].children == [b1.id, b2.id]
    assert s.path_ids(b1.id) == [10, 11]
    assert s.path_ids(b2.id) == [10, 99]


def test_branch_tips():
    s = make()
    a = s.append(ROOT, 10)
    b1 = s.append(a.id, 11)
    b2 = s.append(a.id, 99)
    assert set(s.branch_tips()) == {b1.id, b2.id}


def test_append_to_missing_parent():
    s = make()
    with pytest.raises(KeyError):
        s.append(777, 10)


def test_to_dict_shape():
    s = make()
    a = s.append(ROOT, 10, text="x", logprob=-1.5, rank=2, entropy=3.1)
    d = s.to_dict()
    assert d["prompt_ids"] == [104, 105]
    assert len(d["nodes"]) == 2
    node = d["nodes"][1]
    assert node["id"] == a.id
    assert node["logprob"] == -1.5
    assert node["children"] == []


def test_ids_unique_and_dense():
    s = make()
    ids = [s.append(ROOT, i).id for i in range(5)]
    assert ids == [1, 2, 3, 4, 5]
