import pytest

import logitscope.service as service_mod
from logitscope.capture import CaptureConfig
from logitscope.generate import SamplingParams
from logitscope.model import repo_cache_bytes
from logitscope.service import Service, capture_from, params_from


@pytest.fixture()
def svc(tiny):
    s = Service()
    s.register_bundle("tiny-random", tiny)
    return s


def test_models_info_includes_registered(svc):
    info = {m["name"]: m for m in svc.models_info()}
    assert info["tiny-random"]["loaded"] is True
    assert info["smollm2-135m"]["loaded"] is False
    assert info["smollm2-135m"]["hf_id"] == "HuggingFaceTB/SmolLM2-135M-Instruct"


def test_session_lifecycle(svc):
    s = svc.create_session("tiny-random", "hello")
    assert svc.get_session(s.id) is s
    assert s.prompt_ids == [104, 101, 108, 108, 111]
    svc.delete_session(s.id)
    with pytest.raises(KeyError):
        svc.get_session(s.id)


def test_create_session_requires_loaded_model(svc):
    with pytest.raises(KeyError):
        svc.create_session("smollm2-135m", "hello")


def test_create_session_rejects_empty_prompt(svc):
    with pytest.raises(ValueError):
        svc.create_session("tiny-random", "")


def test_load_progress_lifecycle(svc, tiny, monkeypatch):
    """fake out the download + load so the phase transitions are testable."""
    seen = []

    def fake_prefetch(model_id, on_progress):
        on_progress(50, 100)
        seen.append(dict(svc.progress["smollm2-135m"]))

    monkeypatch.setattr(service_mod, "prefetch_weights", fake_prefetch)
    monkeypatch.setattr(service_mod, "load_model", lambda model_id: tiny)

    svc.load("smollm2-135m")
    assert seen[0] == {"phase": "download", "done_bytes": 50, "total_bytes": 100}
    assert svc.progress["smollm2-135m"]["phase"] == "ready"
    # loading again is a no-op, no progress churn
    svc.load("smollm2-135m")
    assert svc.progress["smollm2-135m"]["phase"] == "ready"


def test_load_failure_reports_error_phase(svc, monkeypatch):
    def boom(model_id, on_progress):
        raise RuntimeError("no network")

    monkeypatch.setattr(service_mod, "prefetch_weights", boom)
    with pytest.raises(RuntimeError):
        svc.load("smollm2-135m")
    p = svc.progress["smollm2-135m"]
    assert p["phase"] == "error"
    assert "no network" in p["error"]


def test_repo_cache_bytes_counts_partials(tmp_path, monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    blobs = tmp_path / "models--org--tiny" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "aaaa").write_bytes(b"x" * 100)
    (blobs / "bbbb.incomplete").write_bytes(b"x" * 40)  # a download mid-flight
    assert repo_cache_bytes("org/tiny") == 140
    assert repo_cache_bytes("org/absent") == 0


def test_params_from_json_dict():
    p = params_from({"temperature": 0.5, "top_k": 10, "seed": 7})
    assert p == SamplingParams(temperature=0.5, top_k=10, seed=7)


def test_params_from_rejects_typos():
    with pytest.raises(ValueError, match="bad sampling params"):
        params_from({"temprature": 0.5})


def test_capture_from_json_dict():
    c = capture_from({"lens": True, "attn_layers": [0, 1], "attn_heads": [2]})
    assert c == CaptureConfig(lens=True, attn_layers=(0, 1), attn_heads=(2,))
    assert capture_from(None) is None


def test_capture_from_rejects_typos():
    with pytest.raises(ValueError, match="bad capture config"):
        capture_from({"lense": True})
