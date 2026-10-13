"""service: the shared state behind the api — loaded models and live sessions.

the app owns exactly one of these. tests (and a future mock mode) inject the
tiny fixture with register_bundle instead of downloading anything.
"""

import os

from .cache import ActivationLRU
from .capture import CaptureConfig
from .generate import SamplingParams
from .model import ModelBundle, load_model, prefetch_weights
from .session import Session, new_session_id

KNOWN_MODELS = {
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    # a local copy (downloaded via the mirror/curl) sidesteps the hf cache's
    # symlink requirement on Windows without Developer Mode.
    "smollm2-local": "F:/models/SmolLM2-135M-Instruct",
}


def params_from(d: dict) -> SamplingParams:
    try:
        return SamplingParams(**d)
    except TypeError as e:
        raise ValueError(f"bad sampling params: {e}") from None


def capture_from(d: dict | None) -> CaptureConfig | None:
    if d is None:
        return None
    d = dict(d)
    for key in ("attn_layers", "attn_heads"):
        if d.get(key) is not None:
            d[key] = tuple(d[key])  # json arrays -> hashable tuples
    try:
        return CaptureConfig(**d)
    except TypeError as e:
        raise ValueError(f"bad capture config: {e}") from None


class Service:
    def __init__(self, cache_bytes: int = 256 * 1024 * 1024):
        self.bundles: dict[str, ModelBundle] = {}
        self.sessions: dict[str, Session] = {}
        # name -> {"phase": download|loading|ready|error, ...}, mutated from the
        # loader thread, read by the progress endpoint. dict ops are atomic
        # enough for a progress bar.
        self.progress: dict[str, dict] = {}
        # keyed by (session_id, node_id, CaptureConfig); values are (meta, frames)
        self.cache = ActivationLRU(cache_bytes)

    # -- models --

    def register_bundle(self, name: str, bundle: ModelBundle) -> None:
        self.bundles[name] = bundle

    def load(self, name: str) -> ModelBundle:
        """load by short name, or any hf causal-lm id as the escape hatch."""
        if name not in self.bundles:
            model_id = KNOWN_MODELS.get(name, name)
            prog = {"phase": "download", "done_bytes": 0, "total_bytes": 0}
            self.progress[name] = prog

            def report(done: int, total: int) -> None:
                prog["done_bytes"] = done
                prog["total_bytes"] = total

            try:
                if not os.path.isdir(model_id):
                    prefetch_weights(model_id, report)
                prog["phase"] = "loading"  # weights -> device, no useful pct
                self.bundles[name] = load_model(model_id)
                prog["phase"] = "ready"
            except Exception as e:
                prog["phase"] = "error"
                prog["error"] = str(e)
                raise
        return self.bundles[name]

    def bundle_for(self, session: Session) -> ModelBundle:
        try:
            return self.bundles[session.model_id]
        except KeyError:
            raise KeyError(f"model {session.model_id} is not loaded") from None

    def models_info(self) -> list[dict]:
        names = sorted(set(KNOWN_MODELS) | set(self.bundles))
        out = []
        for n in names:
            info = {"name": n, "hf_id": KNOWN_MODELS.get(n, n), "loaded": n in self.bundles}
            if n in self.bundles:
                bundle = self.bundles[n]
                cfg = bundle.model.config
                # the ui needs these to build its layer/head pickers
                info["n_layers"] = cfg.num_hidden_layers
                info["n_heads"] = cfg.num_attention_heads
                info["device"] = str(bundle.device)
            out.append(info)
        return out

    # -- sessions --

    def create_session(self, model_name: str, prompt: str) -> Session:
        if model_name not in self.bundles:
            raise KeyError(f"model {model_name} is not loaded")
        if not prompt:
            raise ValueError("empty prompt")
        tok = self.bundles[model_name].tokenizer
        sess = Session(new_session_id(), model_name, prompt, tok.encode(prompt))
        self.sessions[sess.id] = sess
        return sess

    def get_session(self, session_id: str) -> Session:
        try:
            return self.sessions[session_id]
        except KeyError:
            raise KeyError(f"no session {session_id}") from None

    def delete_session(self, session_id: str) -> None:
        self.get_session(session_id)
        del self.sessions[session_id]
