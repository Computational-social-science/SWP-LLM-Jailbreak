"""logit lens: read the unembedding off every layer's hidden state, so you can
watch the final answer crystallize through the stack."""

import torch


@torch.no_grad()
def logit_lens(model, hiddens: list[torch.Tensor], top_k: int = 8) -> list[tuple[list[int], list[float]]]:
    """hiddens: per-layer last-position hidden states (embeddings first), f32 cpu.
    returns one (token_ids, probs) pair per layer, top_k each.

    the final norm is NOT optional: lm_head was trained on normed hidden states,
    so every layer's h goes through model.model.norm before the unembedding.
    skipping it silently rescales the whole distribution (and on a trained model,
    where the norm weight isn't identity, reorders it too). the invariant that
    keeps us honest: the last lens row must equal the model's actual logits.
    """
    head = model.lm_head
    norm = model.model.norm
    device = head.weight.device
    dtype = head.weight.dtype
    out = []
    for h in hiddens:
        z = head(norm(h.to(device=device, dtype=dtype))).float().cpu()
        probs = torch.softmax(z, dim=-1)
        vals, ids = torch.topk(probs, min(top_k, probs.numel()))
        out.append(([int(i) for i in ids], [float(v) for v in vals]))
    return out
