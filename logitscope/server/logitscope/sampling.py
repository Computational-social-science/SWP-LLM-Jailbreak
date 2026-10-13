"""sampling filters, implemented by hand so they can be unit tested by hand.

every filter takes a 1-d float logits tensor and returns a new one. masking is
done with -inf so downstream softmax gives the masked tokens exactly zero mass.
"""

from collections.abc import Iterable

import torch

NEG_INF = float("-inf")


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0, use greedy_token for temp 0")
    return logits / temperature


def apply_repetition_penalty(
    logits: torch.Tensor, seen_ids: Iterable[int], penalty: float
) -> torch.Tensor:
    """ctrl-style: push seen tokens away from being picked again. divide positive
    logits, multiply negative ones — same direction either way."""
    seen = list(set(int(i) for i in seen_ids))
    if penalty == 1.0 or not seen:
        return logits
    out = logits.clone()
    idx = torch.tensor(seen, dtype=torch.long, device=logits.device)
    vals = out[idx]
    out[idx] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return out


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """keep the k largest logits. k <= 0 means off. ties at the kth value are
    all kept (same behavior as hf), so the survivor count can exceed k."""
    if k <= 0 or k >= logits.numel():
        return logits
    kth = torch.topk(logits, k).values[-1]
    return torch.where(logits < kth, NEG_INF, logits)


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """nucleus: keep the smallest prefix of the sorted distribution whose mass
    reaches p. always keeps at least the argmax, so p=0 degrades to greedy."""
    if p >= 1.0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    # a token is cut if the mass *before* it already covers p
    cut = (cum - sorted_probs) >= p
    cut[0] = False
    out = logits.clone()
    out[sorted_idx[cut]] = NEG_INF
    return out


def greedy_token(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits).item())


def sample_token(logits: torch.Tensor, generator: torch.Generator | None = None) -> int:
    """multinomial over softmax(logits). sampling happens on cpu: it's a single
    vocab-sized vector, and cpu generators are seed-stable across devices."""
    probs = torch.softmax(logits.float(), dim=-1).cpu()
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())
