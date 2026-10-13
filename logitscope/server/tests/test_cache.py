import pytest
import torch

from logitscope.cache import ActivationLRU, nbytes_of


def test_nbytes_of_tensor():
    assert nbytes_of(torch.zeros(10, dtype=torch.float16)) == 20
    assert nbytes_of(torch.zeros(4, 4, dtype=torch.float32)) == 64


def test_nbytes_of_containers():
    assert nbytes_of(b"12345") == 5
    assert nbytes_of([b"12", b"345"]) == 5
    assert nbytes_of({"kind": b"1234"}) == 8
    assert nbytes_of(([{"a": 1}], b"xy")) == pytest.approx(11)


def test_nbytes_of_unknown_type_raises():
    with pytest.raises(TypeError):
        nbytes_of(object())


def test_hit_miss_counters():
    c = ActivationLRU(max_bytes=100)
    assert c.get("k") is None
    c.put("k", b"1234")
    assert c.get("k") == b"1234"
    assert c.stats()["hits"] == 1
    assert c.stats()["misses"] == 1


def test_eviction_is_lru_order():
    c = ActivationLRU(max_bytes=10)
    c.put("a", b"aaaa")
    c.put("b", b"bbbb")
    c.get("a")  # touch a so b is now the coldest
    c.put("c", b"cccc")  # 12 bytes total -> evict b
    assert "a" in c
    assert "b" not in c
    assert "c" in c
    assert c.stats()["evictions"] == 1
    assert c.bytes == 8


def test_oversize_value_rejected():
    c = ActivationLRU(max_bytes=10)
    c.put("small", b"1234")
    assert c.put("huge", b"x" * 100) is False
    assert "huge" not in c
    assert "small" in c  # the giant didn't flush the working set


def test_replace_same_key_updates_bytes():
    c = ActivationLRU(max_bytes=100)
    c.put("k", b"1234")
    c.put("k", b"123456")
    assert c.bytes == 6
    assert len(c) == 1


def test_config_keys_are_usable(tiny):
    # frozen dataclass configs hash — the cache key is (session, node, config)
    from logitscope.capture import CaptureConfig

    c = ActivationLRU(max_bytes=100)
    key = ("sess1", 3, CaptureConfig(lens=True))
    c.put(key, b"data")
    assert c.get(("sess1", 3, CaptureConfig(lens=True))) == b"data"
    assert c.get(("sess1", 3, CaptureConfig(lens=False))) is None
