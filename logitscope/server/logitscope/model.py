"""model loading + device selection."""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch


class Tokenizer(Protocol):
    """the two methods we actually need. hf tokenizers already match this."""

    eos_token_id: int | None

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    # f16 on mps. cpu stays f32, half matmuls on cpu are slow and occasionally wrong
    return torch.float16 if device.type == "mps" else torch.float32


@dataclass
class ModelBundle:
    model_id: str
    model: Any  # PreTrainedModel, but keeping the import light
    tokenizer: Tokenizer
    device: torch.device


def repo_cache_bytes(model_id: str) -> int:
    """bytes of this repo already sitting in the hub cache, partial downloads
    included (they land in blobs/ as *.incomplete)."""
    from huggingface_hub import constants

    blobs = Path(constants.HF_HUB_CACHE) / ("models--" + model_id.replace("/", "--")) / "blobs"
    total = 0
    if blobs.is_dir():
        for f in blobs.iterdir():
            try:
                total += f.stat().st_size
            except OSError:
                pass  # a rename raced us, next poll catches it
    return total


# snapshot_download grabs the WHOLE repo by default. smollm2's repo ships
# ~700mb of onnx exports the torch loader never reads — skip that junk.
_SKIP_PATTERNS = ("onnx/*", "*.onnx", "*.gguf")


def prefetch_weights(model_id: str, on_progress: Callable[[int, int], None]) -> None:
    """pull the repo into the hub cache before from_pretrained touches it,
    reporting (done_bytes, total_bytes) as it goes.

    intercepting hf's tqdm bars across its download worker threads is
    miserable, so this doesn't try: a side thread just polls the cache dir
    size twice a second while snapshot_download runs. dumb, version-proof,
    and accurate enough for a progress bar. (done can overshoot a little if
    the cache holds blobs we'd skip today — the ui clamps at 99%.)"""
    from fnmatch import fnmatch

    from huggingface_hub import HfApi, snapshot_download

    try:
        info = HfApi().model_info(model_id, files_metadata=True)
        total = sum(
            s.size or 0
            for s in info.siblings or []
            if not any(fnmatch(s.rfilename, pat) for pat in _SKIP_PATTERNS)
        )
    except Exception:
        total = 0  # offline with a warm cache: snapshot_download still resolves

    stop = threading.Event()

    def poll() -> None:
        while not stop.wait(0.5):
            on_progress(repo_cache_bytes(model_id), total)

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    try:
        snapshot_download(model_id, ignore_patterns=list(_SKIP_PATTERNS))
    finally:
        stop.set()
        poller.join()
    on_progress(repo_cache_bytes(model_id), total)


def load_model(model_id: str, device: torch.device | None = None) -> ModelBundle:
    """load a hf causal lm. eager attention so we can capture attn weights."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or pick_device()
    dtype = pick_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    return ModelBundle(model_id=model_id, model=model, tokenizer=tokenizer, device=device)
