import json

import pytest
from fastapi.testclient import TestClient

from logitscope.app import create_app
from logitscope.engine import extend, fork
from logitscope.generate import SamplingParams
from logitscope.record import session_from_jsonl, session_to_jsonl
from logitscope.service import Service
from logitscope.session import ROOT, Session


def forked_session(tiny) -> Session:
    svc = Service()
    svc.register_bundle("tiny-random", tiny)
    sess = svc.create_session("tiny-random", "save me")
    list(extend(tiny, sess, ROOT, SamplingParams(temperature=0, max_new_tokens=4)))
    # fork the 2nd token onto a different byte, grow the branch a little
    list(fork(tiny, sess, 2, 77, SamplingParams(temperature=0, max_new_tokens=3)))
    return sess


def test_roundtrip_preserves_the_tree(tiny):
    sess = forked_session(tiny)
    back = session_from_jsonl(session_to_jsonl(sess))
    assert back.to_dict() == sess.to_dict()
    # and the rebuilt counter keeps appending after the last id
    n = back.append(back.branch_tips()[0], 9)
    assert n.id == max(sess.nodes) + 1


def test_jsonl_shape(tiny):
    lines = session_to_jsonl(forked_session(tiny)).splitlines()
    head = json.loads(lines[0])
    assert head["kind"] == "session"
    assert head["prompt"] == "save me"
    kinds = {json.loads(line)["kind"] for line in lines[1:]}
    assert kinds == {"node"}
    assert len(lines) == 8  # header + 4 trunk + forced + 2 continuation


def test_import_rejects_missing_header():
    with pytest.raises(ValueError, match="session header"):
        session_from_jsonl('{"kind": "node", "id": 1}\n')


def test_import_rejects_gappy_ids(tiny):
    lines = session_to_jsonl(forked_session(tiny)).splitlines()
    del lines[2]  # drop a middle node: ids no longer dense
    with pytest.raises(ValueError):
        session_from_jsonl("\n".join(lines))


def test_import_rejects_missing_fields():
    head = '{"kind": "session", "id": "x", "model": "m", "prompt": "p", "prompt_ids": [1]}'
    with pytest.raises(ValueError, match="missing field"):
        session_from_jsonl(head + '\n{"kind": "node", "id": 1}\n')


def test_export_import_endpoints(tiny):
    svc = Service()
    svc.register_bundle("tiny-random", tiny)
    client = TestClient(create_app(svc))
    sess = forked_session(tiny)
    svc.sessions[sess.id] = sess

    r = client.get(f"/api/sessions/{sess.id}/export")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/jsonl")

    # import into a fresh server and keep generating on the loaded tree
    svc2 = Service()
    svc2.register_bundle("tiny-random", tiny)
    client2 = TestClient(create_app(svc2))
    r2 = client2.post("/api/sessions/import", content=r.text)
    assert r2.status_code == 200
    assert r2.json() == sess.to_dict()
    loaded = svc2.get_session(sess.id)
    tip = loaded.branch_tips()[0]
    list(extend(tiny, loaded, tip, SamplingParams(temperature=0, max_new_tokens=2)))
    assert len(loaded.nodes) == len(sess.nodes) + 2


def test_import_garbage_400(tiny):
    svc = Service()
    client = TestClient(create_app(svc))
    r = client.post("/api/sessions/import", content="not even json")
    assert r.status_code == 400
    assert "bad session file" in r.json()["detail"]
