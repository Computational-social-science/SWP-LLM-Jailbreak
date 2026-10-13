import numpy as np
import pytest
from fastapi.testclient import TestClient

from logitscope.app import create_app
from logitscope.capture import CaptureConfig
from logitscope.engine import capture_at, extend
from logitscope.generate import SamplingParams
from logitscope.service import Service
from logitscope.session import ROOT
from logitscope.wire import decode_frame


@pytest.fixture()
def rig(tiny):
    svc = Service()
    svc.register_bundle("tiny-random", tiny)
    return TestClient(create_app(svc)), svc


def grow(svc, tiny, n=4):
    sess = svc.create_session("tiny-random", "scrub me")
    bundle = svc.bundles["tiny-random"]
    nodes = [node for node, _ in
             extend(bundle, sess, ROOT, SamplingParams(temperature=0, max_new_tokens=n))]
    return sess, nodes


def test_capture_at_matches_streamed_capture(tiny, rig):
    """recomputing a node's capture must give exactly what streamed by live."""
    _, svc = rig
    sess = svc.create_session("tiny-random", "scrub me")
    bundle = svc.bundles["tiny-random"]
    cfg = CaptureConfig(lens=True, resid_norms=True)
    live = [(node, step) for node, step in
            extend(bundle, sess, ROOT, SamplingParams(temperature=0, max_new_tokens=3),
                   capture=cfg)]
    node, step = live[1]
    re = capture_at(bundle, sess, node.id, cfg)
    assert re.topk_ids == step.capture.topk_ids
    assert re.topk_logits == pytest.approx(step.capture.topk_logits, rel=1e-4)
    assert [ids for ids, _ in re.lens] == [ids for ids, _ in step.capture.lens]
    assert re.forward.resid_norms == pytest.approx(step.capture.forward.resid_norms, rel=1e-4)


def test_capture_at_root_rejected(tiny, rig):
    _, svc = rig
    sess, _ = grow(svc, tiny)
    with pytest.raises(ValueError):
        capture_at(svc.bundles["tiny-random"], sess, ROOT, CaptureConfig())


def scrub_once(client, sess_id, node_id, capture):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "capture", "session_id": sess_id,
                      "node_id": node_id, "capture": capture})
        msg = ws.receive_json()
        if msg["type"] == "error":
            return msg, None
        frames = [decode_frame(ws.receive_bytes()) for _ in msg["frames"]]
        return msg, frames


def test_scrub_miss_then_hit(tiny, rig):
    client, svc = rig
    sess, nodes = grow(svc, tiny)
    target = nodes[2].id
    capture = {"lens": True}

    # count real forwards through the model to prove the hit skips compute
    forwards = 0

    def counter(*a, **k):
        nonlocal forwards
        forwards += 1

    handle = tiny.model.model.embed_tokens.register_forward_pre_hook(counter)
    try:
        first, frames1 = scrub_once(client, sess.id, target, capture)
        second, frames2 = scrub_once(client, sess.id, target, capture)
    finally:
        handle.remove()

    assert first["cached"] is False
    assert second["cached"] is True
    assert forwards == 1  # the second scrub never touched the model
    for (k1, a1), (k2, a2) in zip(frames1, frames2):
        assert k1 == k2
        assert np.array_equal(a1, a2)
    assert svc.cache.stats()["hits"] == 1


def test_scrub_different_config_is_different_entry(tiny, rig):
    client, svc = rig
    sess, nodes = grow(svc, tiny)
    target = nodes[1].id
    a, _ = scrub_once(client, sess.id, target, {"lens": True})
    b, _ = scrub_once(client, sess.id, target, {"lens": True, "resid_norms": True})
    assert a["cached"] is False
    assert b["cached"] is False  # different config, different key
    assert len(svc.cache) == 2


def test_live_stream_warms_the_cache(tiny, rig):
    client, svc = rig
    sess = svc.create_session("tiny-random", "warm me")
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "generate", "session_id": sess.id,
                      "params": {"temperature": 0, "max_new_tokens": 3},
                      "capture": {"lens": True}})
        first_node = None
        while True:
            msg = ws.receive_json()
            if msg["type"] == "done":
                break
            if first_node is None:
                first_node = msg["node"]["id"]
            for _ in msg["frames"]:
                ws.receive_bytes()
    # scrubbing back to a streamed node is a pure cache hit
    msg, _ = scrub_once(client, sess.id, first_node, {"lens": True})
    assert msg["cached"] is True


def test_scrub_unknown_node_errors(tiny, rig):
    client, svc = rig
    sess, _ = grow(svc, tiny)
    msg, _ = scrub_once(client, sess.id, 999, {"lens": True})
    assert msg["type"] == "error"
