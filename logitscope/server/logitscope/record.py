"""session serialization: one jsonl doc per session.

line 1 is the session header, every following line is one node in id order.
jsonl instead of a single json blob so a recorder can append nodes as they
stream, and so you can eyeball a saved session with head/grep.
"""

import json

from .session import ROOT, Session


def session_to_jsonl(session: Session) -> str:
    lines = [
        json.dumps(
            {
                "kind": "session",
                "id": session.id,
                "model": session.model_id,
                "prompt": session.prompt_text,
                "prompt_ids": session.prompt_ids,
                "created": session.created,
            }
        )
    ]
    for k in sorted(session.nodes):
        if k == ROOT:
            continue
        lines.append(json.dumps({"kind": "node", **session.nodes[k].to_dict()}))
    return "\n".join(lines) + "\n"


def session_from_jsonl(text: str) -> Session:
    """rebuild a session, ids and all. children lists come back identical
    because nodes are replayed in id order, which is append order."""
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not records or records[0].get("kind") != "session":
        raise ValueError("first line must be the session header")
    head = records[0]
    try:
        sess = Session(head["id"], head["model"], head["prompt"], head["prompt_ids"])
        sess.created = head.get("created", sess.created)
        for rec in records[1:]:
            if rec.get("kind") != "node":
                raise ValueError(f"unexpected record kind {rec.get('kind')!r}")
            node = sess.append(
                rec["parent"],
                rec["token_id"],
                text=rec.get("text", ""),
                logprob=rec.get("logprob"),
                rank=rec.get("rank"),
                entropy=rec.get("entropy"),
            )
            if node.id != rec["id"]:
                raise ValueError(f"node ids must be dense and in order (got {rec['id']}, expected {node.id})")
    except KeyError as e:
        raise ValueError(f"record missing field {e}") from None
    return sess
