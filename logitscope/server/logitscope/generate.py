"""hand-rolled generation loop.

not model.generate() on purpose: the whole point of this project is seeing
inside every step, so the loop owns the kv cache, the filters, and (later)
the capture hooks, one explicit forward at a time.
"""

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from transformers import DynamicCache

from .capture import CaptureConfig, Recorder, StepCapture, assemble_step
from .model import ModelBundle
from .sampling import (
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    greedy_token,
    sample_token,
)


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0  # 0 means greedy
    top_k: int = 0  # 0 means off
    top_p: float = 1.0  # 1 means off
    repetition_penalty: float = 1.0
    seed: int = 0
    max_new_tokens: int = 32


@dataclass
class StepResult:
    index: int
    token_id: int
    logprob: float  # of the chosen token, under the raw distribution
    rank: int  # 1 = the token was the argmax
    entropy: float  # nats, of the raw distribution
    capture: StepCapture | None = None


def step_stats(raw: torch.Tensor, token: int) -> tuple[float, int, float]:
    """(logprob, rank, entropy) of a chosen token against the *raw* logits.

    raw on purpose: the scope should show what the model believed before the
    sampler mangled it. temperature/top-k/top-p happen after this snapshot.
    """
    logprobs = torch.log_softmax(raw, dim=-1)
    probs = logprobs.exp()
    entropy = float(-(probs * logprobs).sum())
    rank = int((raw > raw[token]).sum().item()) + 1
    return float(logprobs[token]), rank, entropy


@torch.no_grad()
def raw_logits(bundle: ModelBundle, ids: list[int]) -> torch.Tensor:
    """one forward over ids, raw last-position logits as f32."""
    t = torch.tensor([ids], dtype=torch.long, device=bundle.device)
    return bundle.model(input_ids=t).logits[0, -1, :].float()


def _pick(raw: torch.Tensor, context: list[int], params: SamplingParams,
          gen: torch.Generator) -> int:
    logits = raw
    if params.repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, context, params.repetition_penalty)
    if params.temperature == 0:
        return greedy_token(logits)
    logits = apply_temperature(logits, params.temperature)
    if params.top_k > 0:
        logits = apply_top_k(logits, params.top_k)
    if params.top_p < 1.0:
        logits = apply_top_p(logits, params.top_p)
    return sample_token(logits, gen)


@torch.no_grad()
def generate_stream(
    bundle: ModelBundle,
    prompt_ids: list[int],
    params: SamplingParams,
    capture: CaptureConfig | None = None,
) -> Iterator[StepResult]:
    """yield one StepResult per generated token.

    the prompt is prefilled in a single forward, then each step feeds exactly
    one token back in with the kv cache carrying the rest of the context.
    with a capture config, forward hooks record what each pass saw; the
    capture on step i belongs to the forward that *predicted* token i.
    """
    if not prompt_ids:
        raise ValueError("empty prompt")
    model, device = bundle.model, bundle.device
    eos = bundle.tokenizer.eos_token_id
    gen = torch.Generator("cpu").manual_seed(params.seed)
    cache = DynamicCache()
    context = list(prompt_ids)

    rec_cm = Recorder(model, capture) if capture is not None else contextlib.nullcontext()
    with rec_cm as rec:
        ids = torch.tensor([context], dtype=torch.long, device=device)
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)

        for i in range(params.max_new_tokens):
            raw = out.logits[0, -1, :].float()
            token = _pick(raw, context, params, gen)
            logprob, rank, entropy = step_stats(raw, token)
            step_capture = None
            if capture is not None:
                step_capture = assemble_step(model, raw, rec, capture)
            yield StepResult(index=i, token_id=token, logprob=logprob, rank=rank,
                             entropy=entropy, capture=step_capture)
            context.append(token)
            if eos is not None and token == eos:
                return
            step = torch.tensor([[token]], dtype=torch.long, device=device)
            out = model(input_ids=step, past_key_values=cache, use_cache=True)
