"""activation lru.

scrubbing back through history in the ui asks for the same step captures over
and over, and recomputing a forward pass per hover would be miserable. so
captures live in a byte-budgeted lru keyed by (session, node, capture config).
values are whatever the caller wants to stash (we use the already-encoded wire
frames — no point caching tensors just to re-encode them every scrub).
"""

from collections import OrderedDict

import torch


def nbytes_of(obj) -> int:
    """rough-but-honest payload size. counts the stuff that actually weighs
    something; fixed per-object python overhead is noise at our sizes."""
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return len(obj)
    if isinstance(obj, str):
        return len(obj.encode("utf-8"))
    if isinstance(obj, dict):
        return sum(nbytes_of(k) + nbytes_of(v) for k, v in obj.items())
    if isinstance(obj, (list, tuple, set)):
        return sum(nbytes_of(x) for x in obj)
    if isinstance(obj, (int, float, bool)) or obj is None:
        return 8
    raise TypeError(f"don't know the size of {type(obj).__name__}")


class ActivationLRU:
    def __init__(self, max_bytes: int):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._entries: OrderedDict = OrderedDict()  # key -> (value, nbytes)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key) -> bool:
        return key in self._entries

    def get(self, key):
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        self._entries.move_to_end(key)
        return entry[0]

    def put(self, key, value, nbytes: int | None = None) -> bool:
        """insert (or refresh) an entry. returns False if the value alone is
        bigger than the whole budget — those are not admitted, one giant
        attention map shouldn't flush everything else."""
        size = nbytes_of(value) if nbytes is None else nbytes
        if size > self.max_bytes:
            return False
        old = self._entries.pop(key, None)
        if old is not None:
            self.bytes -= old[1]
        self._entries[key] = (value, size)
        self.bytes += size
        while self.bytes > self.max_bytes:
            _, (_, evicted_size) = self._entries.popitem(last=False)
            self.bytes -= evicted_size
            self.evictions += 1
        return True

    def stats(self) -> dict:
        return {
            "entries": len(self._entries),
            "bytes": self.bytes,
            "max_bytes": self.max_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }
