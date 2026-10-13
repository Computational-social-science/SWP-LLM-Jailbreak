import pytest
from fastapi.testclient import TestClient

from logitscope.app import create_app
from logitscope.engine import extend
from logitscope.generate import SamplingParams
from logitscope.service import Service
from logitscope.session import ROOT


@pytest.fixture()
def rig(tiny):
    svc = Service()
    svc.register_bundle("tiny-random", tiny)
    return TestClient(create_app(svc)), svc


def test_health(rig):
    client, _ = rig
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_models_lists_known_and_loaded(rig):
    client, _ = rig
    r = client.get("/api/models")
    assert r.status_code == 200
    by_name = {m["name"]: m for m in r.json()}
    assert by_name["tiny-random"]["loaded"] is True
    assert by_name["smollm2-135m"]["loaded"] is False
    assert by_name["qwen3-0.6b"]["hf_id"] == "Qwen/Qwen3-0.6B"


def test_models_info_carries_shape_only_when_loaded(rig):
    client, _ = rig
    by_name = {m["name"]: m for m in client.get("/api/models").json()}
    assert by_name["tiny-random"]["n_layers"] == 2
    assert by_name["tiny-random"]["n_heads"] == 4
    assert by_name["tiny-random"]["device"] == "cpu"
    assert "n_layers" not in by_name["smollm2-135m"]  # not loaded, shape unknown


def test_decode_tokens_one_id_at_a_time(rig):
    client, _ = rig
    r = client.post("/api/models/tiny-random/decode", json={"ids": [104, 105, 33]})
    assert r.status_code == 200
    assert r.json()["texts"] == ["h", "i", "!"]


def test_decode_empty_ids(rig):
    client, _ = rig
    assert client.post("/api/models/tiny-random/decode", json={"ids": []}).json() == {"texts": []}


def test_decode_unloaded_model_404(rig):
    client, _ = rig
    r = client.post("/api/models/smollm2-135m/decode", json={"ids": [1]})
    assert r.status_code == 404


def test_load_already_registered_model(rig):
    client, _ = rig
    r = client.post("/api/models/load", json={"name": "tiny-random"})
    assert r.status_code == 200
    body = r.json()
    assert body["loaded"] is True
    assert body["device"] == "cpu"


def test_load_progress_endpoint(rig):
    client, svc = rig
    assert client.get("/api/models/progress").json() == {}
    svc.progress["qwen3-0.6b"] = {"phase": "download", "done_bytes": 1, "total_bytes": 2}
    r = client.get("/api/models/progress")
    assert r.json()["qwen3-0.6b"]["phase"] == "download"


def test_session_crud_roundtrip(rig):
    client, _ = rig
    r = client.post("/api/sessions", json={"model": "tiny-random", "prompt": "hi there"})
    assert r.status_code == 200
    sid = r.json()["id"]
    assert r.json()["prompt"] == "hi there"

    listed = client.get("/api/sessions").json()
    assert [s["id"] for s in listed] == [sid]

    got = client.get(f"/api/sessions/{sid}")
    assert got.status_code == 200
    assert got.json()["prompt_ids"] == [ord(c) for c in "hi there"]

    assert client.delete(f"/api/sessions/{sid}").json() == {"ok": True}
    assert client.get(f"/api/sessions/{sid}").status_code == 404
    assert client.delete(f"/api/sessions/{sid}").status_code == 404


def test_create_session_bad_model_400(rig):
    client, _ = rig
    r = client.post("/api/sessions", json={"model": "smollm2-135m", "prompt": "x"})
    assert r.status_code == 400
    assert "not loaded" in r.json()["detail"]


def test_create_session_empty_prompt_400(rig):
    client, _ = rig
    r = client.post("/api/sessions", json={"model": "tiny-random", "prompt": ""})
    assert r.status_code == 400


def test_session_tree_shows_forks(rig, tiny):
    client, svc = rig
    sid = client.post("/api/sessions",
                      json={"model": "tiny-random", "prompt": "tree me"}).json()["id"]
    sess = svc.get_session(sid)
    list(extend(tiny, sess, ROOT, SamplingParams(temperature=0, max_new_tokens=3)))
    list(extend(tiny, sess, ROOT, SamplingParams(seed=5, max_new_tokens=2)))

    tree = client.get(f"/api/sessions/{sid}").json()
    assert len(tree["nodes"]) == 6  # root + 3 + 2
    root = tree["nodes"][0]
    assert len(root["children"]) == 2  # two branches off the root


def test_cache_stats_endpoint(rig):
    client, _ = rig
    r = client.get("/api/cache/stats")
    assert r.status_code == 200
    stats = r.json()
    assert stats["entries"] == 0
    assert stats["max_bytes"] > 0


def test_missing_body_is_422(rig):
    client, _ = rig
    assert client.post("/api/sessions", json={"model": "tiny-random"}).status_code == 422
