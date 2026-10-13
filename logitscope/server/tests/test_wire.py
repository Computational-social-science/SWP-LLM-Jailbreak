import gzip

import numpy as np
import pytest
import torch

from logitscope.wire import (
    KIND_ATTN,
    KIND_LENS_PROBS,
    KIND_TOPK_IDS,
    KIND_TOPK_LOGITS,
    MAGIC,
    decode_frame,
    encode_frame,
)


def roundtrip(kind, arr):
    return decode_frame(encode_frame(kind, arr))


def test_kind_and_shape_survive():
    kind, arr = roundtrip(KIND_ATTN, torch.zeros(4, 3, 7))
    assert kind == KIND_ATTN
    assert arr.shape == (4, 3, 7)
    assert arr.dtype == np.float16


def test_ids_roundtrip_exact():
    ids = np.array([0, 1, 255, 65535, 4_000_000_000], dtype=np.uint64)
    _, arr = roundtrip(KIND_TOPK_IDS, ids)
    assert arr.tolist() == ids.tolist()
    assert arr.dtype == np.uint32


def test_f16_values_roundtrip_exact():
    # values already representable in f16 come back bit-identical
    x = np.array([0.0, 0.5, -2.0, 1024.0, 6.104e-5], dtype=np.float16)
    _, arr = roundtrip(KIND_LENS_PROBS, x)
    assert np.array_equal(arr, x)


def test_f16_quantization_error_bound():
    """f16 has a 10-bit mantissa: relative error <= 2^-11 for normal values.
    logits live comfortably inside +-100, nowhere near f16 limits."""
    rng = np.random.default_rng(7)
    x = rng.uniform(-100, 100, size=4096).astype(np.float32)
    _, arr = roundtrip(KIND_TOPK_LOGITS, x)
    rel = np.abs(arr.astype(np.float32) - x) / np.abs(x)
    assert float(rel.max()) <= 2**-11


def test_probs_error_absolute():
    # probabilities in [0,1]: absolute error stays tiny too
    rng = np.random.default_rng(8)
    p = rng.uniform(0, 1, size=1024).astype(np.float32)
    _, arr = roundtrip(KIND_LENS_PROBS, p)
    assert float(np.abs(arr.astype(np.float32) - p).max()) < 1e-3


def test_scalar_and_empty():
    _, s = roundtrip(KIND_TOPK_LOGITS, np.float32(3.5))
    assert s.shape == ()
    assert float(s) == 3.5
    _, e = roundtrip(KIND_TOPK_LOGITS, np.zeros(0, dtype=np.float32))
    assert e.shape == (0,)


def test_torch_tensor_accepted_from_f64():
    _, arr = roundtrip(KIND_TOPK_LOGITS, torch.arange(6, dtype=torch.float64).reshape(2, 3))
    assert arr.shape == (2, 3)
    assert arr.dtype == np.float16


def test_bad_magic_rejected():
    buf = encode_frame(KIND_ATTN, np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError, match="magic"):
        decode_frame(b"XX" + buf[2:])


def test_bad_version_rejected():
    buf = bytearray(encode_frame(KIND_ATTN, np.zeros(3, dtype=np.float32)))
    buf[2] = 99
    with pytest.raises(ValueError, match="version"):
        decode_frame(bytes(buf))


def test_truncated_frame_rejected():
    buf = encode_frame(KIND_ATTN, np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        decode_frame(buf[:5])


def test_size_mismatch_rejected():
    """shape says 4 floats but payload carries 3 — must not silently misread."""
    good = encode_frame(KIND_ATTN, np.zeros(3, dtype=np.float32))
    header = good[:6]
    tampered = header[:6] + b"\x04\x00\x00\x00" + gzip.compress(np.zeros(3, "<f2").tobytes())
    with pytest.raises(ValueError, match="payload"):
        decode_frame(tampered)


def test_bool_rejected():
    with pytest.raises(TypeError):
        encode_frame(KIND_ATTN, np.array([True, False]))


def test_header_is_small():
    # 6 byte header + one u32 dim: overhead shouldn't creep up silently
    buf = encode_frame(KIND_TOPK_LOGITS, np.zeros(1, dtype=np.float16))
    assert buf[:2] == MAGIC
    assert len(buf) < 40
