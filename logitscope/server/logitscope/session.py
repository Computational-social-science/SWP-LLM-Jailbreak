"""session tree.

a generation is not a list, it's a tree: every token knows its parent, and a
fork is a sibling branch growing out of the same prefix. node 0 is the root
anchor that stands in for the prompt itself.
"""

import itertools
import time
import uuid
from dataclasses import dataclass, field

ROOT = 0


@dataclass
class TokenNode:
    id: int
    parent: int | None  # None only for the root
    token_id: int | None  # None only for the root
    text: str = ""
    logprob: float | None = None
    rank: int | None = None
    entropy: float | None = None
    children: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "parent": self.parent,
            "token_id": self.token_id,
            "text": self.text,
            "logprob": self.logprob,
            "rank": self.rank,
            "entropy": self.entropy,
            "children": list(self.children),
        }


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


class Session:
    def __init__(self, session_id: str, model_id: str, prompt_text: str, prompt_ids: list[int]):
        self.id = session_id
        self.model_id = model_id
        self.prompt_text = prompt_text
        self.prompt_ids = list(prompt_ids)
        self.created = time.time()
        self.nodes: dict[int, TokenNode] = {ROOT: TokenNode(id=ROOT, parent=None, token_id=None)}
        self._ids = itertools.count(1)

    def append(self, parent_id: int, token_id: int, *, text: str = "",
               logprob: float | None = None, rank: int | None = None,
               entropy: float | None = None) -> TokenNode:
        if parent_id not in self.nodes:
            raise KeyError(f"no node {parent_id} in session {self.id}")
        node = TokenNode(id=next(self._ids), parent=parent_id, token_id=token_id,
                         text=text, logprob=logprob, rank=rank, entropy=entropy)
        self.nodes[node.id] = node
        self.nodes[parent_id].children.append(node.id)
        return node

    def path_ids(self, node_id: int) -> list[int]:
        """token ids from just after the prompt down to node_id, inclusive."""
        out: list[int] = []
        cur = node_id
        while cur != ROOT:
            n = self.nodes[cur]
            out.append(n.token_id)
            cur = n.parent
        return out[::-1]

    def context_ids(self, node_id: int) -> list[int]:
        """full model context for continuing below node_id: prompt + path."""
        return self.prompt_ids + self.path_ids(node_id)

    def branch_tips(self) -> list[int]:
        """leaves of the tree — one per branch."""
        return [n.id for n in self.nodes.values() if not n.children and n.id != ROOT]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "model": self.model_id,
            "prompt": self.prompt_text,
            "prompt_ids": list(self.prompt_ids),
            "created": self.created,
            "nodes": [self.nodes[k].to_dict() for k in sorted(self.nodes)],
        }
