"""generate golden frames for the js decoder tests, using the real python
encoder so the two ends can never drift apart silently.

run from web/:  ../server/.venv/bin/python scripts/gen_golden.py > src/wire.golden.json
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "server"))

from logitscope import wire  # noqa: E402


def clean(x: float) -> float | str:
    # json has no nan/inf, ship them as strings and undo it in the test
    if np.isnan(x):
        return "NaN"
    if np.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    return float(x)


def case(name: str, kind: int, arr) -> dict:
    frame = wire.encode_frame(kind, arr)
    arr = np.asarray(arr)
    if arr.dtype.kind == "f":
        # what the wire actually carries: f16-quantized values
        expected = arr.astype("<f2").astype("f4")
    else:
        expected = arr.astype("<u4")
    return {
        "name": name,
        "kind": kind,
        "shape": list(arr.shape),
        "b64": base64.b64encode(frame).decode(),
        "expected": [clean(v) for v in expected.ravel().tolist()],
    }


cases = [
    case("topk_ids", wire.KIND_TOPK_IDS,
         np.array([0, 1, 41, 65535, 4294967295], dtype=np.uint32)),
    case("topk_logits", wire.KIND_TOPK_LOGITS,
         np.array([12.375, -3.140625, 0.0, -0.0, 1e-7, 65504.0], dtype=np.float32)),
    case("lens_probs_2d", wire.KIND_LENS_PROBS,
         np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4)),
    case("attn_3d", wire.KIND_ATTN,
         (np.arange(24, dtype=np.float32) / 24).reshape(2, 3, 4)),
    case("f16_overflow_is_inf", wire.KIND_TOPK_LOGITS,
         np.array([1e6, -1e6, float("nan")], dtype=np.float32)),
    case("zero_dim", wire.KIND_RESID_NORMS, np.float32(3.140625)),
]

print(json.dumps(cases, indent=1))
