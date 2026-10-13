"""binary ws frames.

json is fine for control messages, but tensors as json arrays are a joke —
every float becomes ~13 bytes of text. tensor payloads ride in binary frames:

    "LS" | version u8 | kind u8 | dtype u8 | ndim u8 | shape u32 x ndim | gzip(data)

everything little-endian. floats travel as f16: the ui draws bars and heatmaps,
it does not need 24 bits of mantissa. ids travel as u32.
"""

import gzip
import struct

import numpy as np
import torch

MAGIC = b"LS"
VERSION = 1

# frame kinds
KIND_TOPK_IDS = 1
KIND_TOPK_LOGITS = 2
KIND_LENS_IDS = 3
KIND_LENS_PROBS = 4
KIND_ATTN = 5
KIND_RESID_NORMS = 6
KIND_HIDDEN = 7

DT_F16 = 1
DT_F32 = 2
DT_U32 = 3

_DTYPES = {
    DT_F16: np.dtype("<f2"),
    DT_F32: np.dtype("<f4"),
    DT_U32: np.dtype("<u4"),
}
_HEADER = struct.Struct("<2sBBBB")


def _as_wire_array(arr) -> np.ndarray:
    """normalize input to the on-wire dtype: floats -> f16, ints -> u32."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.dtype.kind == "f":
        return arr.astype("<f2")
    if arr.dtype.kind in "iu":
        return arr.astype("<u4")
    raise TypeError(f"can't send dtype {arr.dtype} over the wire")


def encode_frame(kind: int, arr) -> bytes:
    if not 0 < kind < 256:
        raise ValueError(f"bad frame kind {kind}")
    data = _as_wire_array(arr)
    code = DT_F16 if data.dtype == np.dtype("<f2") else DT_U32
    header = _HEADER.pack(MAGIC, VERSION, kind, code, data.ndim)
    shape = struct.pack(f"<{data.ndim}I", *data.shape)
    # mtime=0 keeps encode_frame deterministic — same tensor, same bytes.
    # otherwise the gzip header sneaks a timestamp in and golden fixtures churn.
    return header + shape + gzip.compress(data.tobytes(), mtime=0)


def frames_for_step(cap) -> tuple[list[dict], list[bytes]]:
    """binary frames for a StepCapture plus their json descriptions, in matching
    order. the token control message carries the descriptions so the client
    knows what the next N binary frames are without guessing."""
    if cap is None:
        return [], []
    meta: list[dict] = []
    frames: list[bytes] = []

    def add(desc: dict, kind: int, arr) -> None:
        meta.append(desc)
        frames.append(encode_frame(kind, arr))

    add({"kind": "topk_ids"}, KIND_TOPK_IDS, np.asarray(cap.topk_ids, dtype=np.uint32))
    add({"kind": "topk_logits"}, KIND_TOPK_LOGITS, np.asarray(cap.topk_logits, dtype=np.float32))
    if cap.lens is not None:
        add({"kind": "lens_ids"}, KIND_LENS_IDS,
            np.asarray([ids for ids, _ in cap.lens], dtype=np.uint32))
        add({"kind": "lens_probs"}, KIND_LENS_PROBS,
            np.asarray([ps for _, ps in cap.lens], dtype=np.float32))
    fc = cap.forward
    if fc is not None:
        if fc.resid_norms is not None:
            add({"kind": "resid_norms"}, KIND_RESID_NORMS,
                np.asarray(fc.resid_norms, dtype=np.float32))
        if fc.hidden is not None:
            add({"kind": "hidden"}, KIND_HIDDEN, torch.stack(fc.hidden))
        if fc.attn is not None:
            for layer in sorted(fc.attn):
                add({"kind": "attn", "layer": layer}, KIND_ATTN, fc.attn[layer])
    return meta, frames


def decode_frame(buf: bytes) -> tuple[int, np.ndarray]:
    if len(buf) < _HEADER.size:
        raise ValueError("frame shorter than header")
    magic, version, kind, dtype_code, ndim = _HEADER.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if version != VERSION:
        raise ValueError(f"unknown version {version}")
    if dtype_code not in _DTYPES:
        raise ValueError(f"unknown dtype code {dtype_code}")
    dtype = _DTYPES[dtype_code]
    off = _HEADER.size
    if len(buf) < off + 4 * ndim:
        raise ValueError("frame truncated in shape")
    shape = struct.unpack_from(f"<{ndim}I", buf, off)
    off += 4 * ndim
    data = gzip.decompress(buf[off:])
    expect = int(np.prod(shape)) * dtype.itemsize if ndim else dtype.itemsize
    if len(data) != expect:
        raise ValueError(f"payload is {len(data)} bytes, shape says {expect}")
    return kind, np.frombuffer(data, dtype=dtype).reshape(shape)
