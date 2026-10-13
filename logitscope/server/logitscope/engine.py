"""engine: grows a session tree by running the generation loop.

two ways to branch, matching what the ui needs:
  - extend(): keep generating below any node. calling it on a node that already
    has children just grows a sibling branch — that IS the params-fork.
  - fork(): "what if the model had said X instead" — replace the token at a
    node with a forced choice, as a sibling, then keep generating from there.
"""

from collections.abc import Iterator

from .capture import CaptureConfig, Recorder, StepCapture, assemble_step
from .generate import SamplingParams, StepResult, generate_stream, raw_logits, step_stats
from .model import ModelBundle
from .session import ROOT, Session, TokenNode


def _decode(bundle: ModelBundle, token_id: int) -> str:
    return bundle.tokenizer.decode([token_id])


def extend(
    bundle: ModelBundle,
    session: Session,
    from_node: int,
    params: SamplingParams,
    capture: CaptureConfig | None = None,
) -> Iterator[tuple[TokenNode, StepResult]]:
    """generate below from_node, appending one child per token."""
    ctx = session.context_ids(from_node)
    parent = from_node
    for step in generate_stream(bundle, ctx, params, capture=capture):
        node = session.append(parent, step.token_id, text=_decode(bundle, step.token_id),
                              logprob=step.logprob, rank=step.rank, entropy=step.entropy)
        parent = node.id
        yield node, step


def fork(
    bundle: ModelBundle,
    session: Session,
    at_node: int,
    forced_token: int,
    params: SamplingParams,
    capture: CaptureConfig | None = None,
) -> Iterator[tuple[TokenNode, StepResult]]:
    """sibling branch: same prefix as at_node, but the token at that position
    is forced_token instead of what the model picked. the forced token still
    gets honest stats (its logprob/rank under the model at that prefix), then
    generation continues normally below it."""
    target = session.nodes[at_node]
    if target.parent is None:
        raise ValueError("can't fork the root anchor")
    prefix_node = target.parent
    ctx = session.context_ids(prefix_node)

    raw = raw_logits(bundle, ctx)
    if not 0 <= forced_token < raw.numel():
        raise ValueError(f"forced token {forced_token} out of vocab")
    logprob, rank, entropy = step_stats(raw, forced_token)
    forced = session.append(prefix_node, forced_token, text=_decode(bundle, forced_token),
                            logprob=logprob, rank=rank, entropy=entropy)
    yield forced, StepResult(index=0, token_id=forced_token, logprob=logprob,
                             rank=rank, entropy=entropy)

    remaining = params.max_new_tokens - 1
    if remaining > 0:
        cont = SamplingParams(temperature=params.temperature, top_k=params.top_k,
                              top_p=params.top_p, repetition_penalty=params.repetition_penalty,
                              seed=params.seed, max_new_tokens=remaining)
        yield from extend(bundle, session, forced.id, cont, capture=capture)


def capture_at(
    bundle: ModelBundle,
    session: Session,
    node_id: int,
    cfg: CaptureConfig,
) -> StepCapture:
    """recompute the capture for one existing node with a single forward.

    the capture that 'belongs' to a node is from the forward that predicted it,
    so the context is everything up to the node's parent. this is the recompute
    path behind the activation cache — a scrub miss lands here."""
    node = session.nodes[node_id]
    if node.parent is None:
        raise ValueError("the root anchor has no capture")
    ctx = session.context_ids(node.parent)
    with Recorder(bundle.model, cfg) as rec:
        raw = raw_logits(bundle, ctx)
        return assemble_step(bundle.model, raw, rec, cfg)


def capture_prompt(
    bundle: ModelBundle,
    session: Session,
    cfg: CaptureConfig,
) -> StepCapture:
    """capture over the prompt tokens only — the prefill forward with no
    generated context. powers the input-only compare view: two sentences'
    tokenization + attention, side by side, before anything is generated."""
    ctx = session.prompt_ids
    with Recorder(bundle.model, cfg) as rec:
        raw = raw_logits(bundle, ctx)
        return assemble_step(bundle.model, raw, rec, cfg)
