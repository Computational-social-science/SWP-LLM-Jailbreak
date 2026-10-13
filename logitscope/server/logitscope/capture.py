"""forward-hook capture.

a Recorder hangs real forward hooks on the model's decoder layers. after each
forward, harvest() pops whatever the hooks saw for that pass. everything is
detached and moved to cpu immediately so nothing pins the mps graph.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CaptureConfig:
    topk_logits: int = 32
    hidden: bool = False  # per-layer last-position hidden states (embeddings first)
    resid_norms: bool = False  # l2 norm of the residual stream per layer
    lens: bool = False  # logit lens rows per layer (needs hidden hooks)
    lens_topk: int = 8
    # attention is capture-on-demand ONLY: you name the layers you want, there is
    # deliberately no "give me everything" switch. keeping every layer x head map
    # is layers*heads*ctx^2 floats per step — even a 135m model (30 layers, 9
    # heads) at 1k context is >1 GiB per step in f32. ask for what you're looking at.
    attn_layers: tuple[int, ...] | None = None  # None = attention capture off
    attn_heads: tuple[int, ...] | None = None  # None = all heads of the selected layers


@dataclass
class ForwardCapture:
    # hidden[0] is the embedding output, hidden[i] is decoder layer i-1's output,
    # all at the last position only, f32 on cpu
    hidden: list[torch.Tensor] | None = None
    resid_norms: list[float] | None = None
    # layer -> (heads, q_len, kv_len), f16 on cpu. q_len is 1 for decode steps
    # and prompt_len for the prefill; rows accumulate client-side.
    attn: dict[int, torch.Tensor] | None = None


@dataclass
class StepCapture:
    topk_ids: list[int]
    topk_logits: list[float]
    forward: ForwardCapture | None = None
    # one (token_ids, probs) pair per layer, embeddings first
    lens: list[tuple[list[int], list[float]]] | None = None


def decoder_layers(model):
    # llama-family layout (llama, smollm, qwen): model.model.layers
    return model.model.layers


def embedding_module(model):
    return model.model.embed_tokens


class Recorder:
    """context manager. hooks attach on enter, detach on exit — never leave
    hooks on a shared model."""

    def __init__(self, model, cfg: CaptureConfig):
        self.model = model
        self.cfg = cfg
        self._handles: list = []
        self._hidden: dict[int, torch.Tensor] = {}
        self._attn: dict[int, torch.Tensor] = {}

    def __enter__(self):
        layers = decoder_layers(self.model)
        if self.cfg.hidden or self.cfg.resid_norms or self.cfg.lens:
            emb = embedding_module(self.model)
            self._handles.append(emb.register_forward_hook(self._hidden_hook(0)))
            for i, layer in enumerate(layers):
                self._handles.append(layer.register_forward_hook(self._hidden_hook(i + 1)))
        if self.cfg.attn_layers is not None:
            for i in self.cfg.attn_layers:
                if not 0 <= i < len(layers):
                    raise ValueError(f"attn layer {i} out of range (model has {len(layers)})")
                self._handles.append(
                    layers[i].self_attn.register_forward_hook(self._attn_hook(i))
                )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def _hidden_hook(self, idx: int):
        def fn(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            self._hidden[idx] = h[0, -1, :].detach().float().cpu()

        return fn

    def _attn_hook(self, idx: int):
        def fn(module, args, output):
            # eager attention returns (attn_output, attn_weights); sdpa/flash
            # never materialize the weights, which is why we force eager
            weights = output[1]
            if weights is None:
                raise RuntimeError(
                    "attention weights are None — model not running eager attention?"
                )
            w = weights[0]  # (heads, q_len, kv_len)
            if self.cfg.attn_heads is not None:
                w = w[list(self.cfg.attn_heads)]
            self._attn[idx] = w.detach().to(torch.float16).cpu()

        return fn

    def harvest(self) -> ForwardCapture:
        """pop the capture for the forward that just ran."""
        cap = ForwardCapture()
        if self._hidden:
            ordered = [self._hidden[i] for i in sorted(self._hidden)]
            if self.cfg.hidden or self.cfg.lens:
                cap.hidden = ordered
            if self.cfg.resid_norms:
                cap.resid_norms = [float(torch.linalg.vector_norm(h)) for h in ordered]
            self._hidden = {}
        if self._attn:
            cap.attn = self._attn
            self._attn = {}
        return cap


def topk_snapshot(raw: torch.Tensor, k: int) -> tuple[list[int], list[float]]:
    """top-k of the raw logits — 'what it almost said'."""
    k = min(k, raw.numel())
    values, ids = torch.topk(raw, k)
    return [int(i) for i in ids], [float(v) for v in values]


def assemble_step(model, raw: torch.Tensor, rec: Recorder, cfg: CaptureConfig) -> StepCapture:
    """turn one forward's raw logits + harvested hooks into a StepCapture.
    shared by the generation loop and single-forward recompute (scrubbing)."""
    from .lens import logit_lens

    ids_k, vals_k = topk_snapshot(raw, cfg.topk_logits)
    fc = rec.harvest()
    lens_rows = None
    if cfg.lens and fc.hidden is not None:
        lens_rows = logit_lens(model, fc.hidden, cfg.lens_topk)
        if not cfg.hidden:
            fc.hidden = None  # hooks kept them only for the lens
    return StepCapture(topk_ids=ids_k, topk_logits=vals_k, forward=fc, lens=lens_rows)
