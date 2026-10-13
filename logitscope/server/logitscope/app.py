"""fastapi app. rest for models/sessions, websocket for streaming generation.

ws protocol: client sends json control messages; per generated token the server
sends one json "token" message describing the node and the binary frames that
follow, then that many binary frames, then a final "done".
"""

import asyncio

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from . import __version__
from .capture import CaptureConfig
from .engine import capture_at, capture_prompt, extend, fork
from .record import session_from_jsonl, session_to_jsonl
from .service import Service, capture_from, params_from
from .session import ROOT
from .wire import frames_for_step


class LoadModelBody(BaseModel):
    name: str


class CreateSessionBody(BaseModel):
    model: str
    prompt: str


class DecodeBody(BaseModel):
    ids: list[int]


def create_app(service: Service | None = None) -> FastAPI:
    app = FastAPI(title="logitscope", version=__version__)
    svc = service or Service()
    app.state.service = svc

    def session_or_404(session_id: str):
        try:
            return svc.get_session(session_id)
        except KeyError as e:
            raise HTTPException(404, str(e)) from None

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__}

    @app.get("/api/models")
    def models():
        return svc.models_info()

    @app.post("/api/models/load")
    async def load_model(body: LoadModelBody):
        # downloading weights can take minutes — keep the loop free
        try:
            bundle = await asyncio.to_thread(svc.load, body.name)
        except Exception as e:  # hf raises half a dozen flavors for a bad id
            raise HTTPException(400, f"could not load {body.name!r}: {e}") from None
        return {"name": body.name, "hf_id": bundle.model_id, "loaded": True,
                "device": str(bundle.device)}

    @app.get("/api/models/progress")
    def load_progress():
        # polled by the ui while a load is in flight
        return svc.progress

    @app.post("/api/models/{name}/decode")
    def decode_tokens(name: str, body: DecodeBody):
        if name not in svc.bundles:
            raise HTTPException(404, f"model {name} is not loaded")
        tok = svc.bundles[name].tokenizer
        # one id at a time on purpose: the ui labels individual tokens, and a
        # joint decode would merge bpe pieces into one string
        return {"texts": [tok.decode([i]) for i in body.ids]}

    @app.post("/api/sessions")
    def create_session(body: CreateSessionBody):
        try:
            sess = svc.create_session(body.model, body.prompt)
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e)) from None
        return sess.to_dict()

    @app.get("/api/sessions")
    def list_sessions():
        return [
            {"id": s.id, "model": s.model_id, "prompt": s.prompt_text,
             "nodes": len(s.nodes), "created": s.created}
            for s in svc.sessions.values()
        ]

    @app.post("/api/sessions/import")
    async def import_session(request: Request):
        text = (await request.body()).decode("utf-8", errors="replace")
        try:
            sess = session_from_jsonl(text)
        except ValueError as e:
            raise HTTPException(400, f"bad session file: {e}") from None
        # same id in = same id out, so re-importing a file just overwrites
        svc.sessions[sess.id] = sess
        return sess.to_dict()

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str):
        return session_or_404(session_id).to_dict()

    @app.get("/api/sessions/{session_id}/export")
    def export_session(session_id: str):
        return PlainTextResponse(
            session_to_jsonl(session_or_404(session_id)),
            media_type="application/jsonl",
        )

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str):
        session_or_404(session_id)
        svc.delete_session(session_id)
        return {"ok": True}

    @app.get("/api/cache/stats")
    def cache_stats():
        return svc.cache.stats()

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            while True:
                msg = await sock.receive_json()
                try:
                    await _dispatch(svc, sock, msg)
                except (KeyError, ValueError) as e:
                    await sock.send_json({"type": "error",
                                          "session_id": msg.get("session_id"),
                                          "error": str(e)})
        except WebSocketDisconnect:
            pass

    return app


async def _dispatch(svc: Service, sock: WebSocket, msg: dict) -> None:
    kind = msg.get("type")
    if kind not in ("generate", "fork", "capture", "capture_prompt"):
        await sock.send_json({"type": "error", "error": f"unknown message type {kind!r}"})
        return
    session = svc.get_session(msg["session_id"])
    bundle = svc.bundle_for(session)
    cap = capture_from(msg.get("capture"))
    if kind == "capture":
        await _scrub(svc, sock, session, bundle, msg["node_id"], cap or CaptureConfig())
        return
    if kind == "capture_prompt":
        await _scrub_prompt(svc, sock, session, bundle, cap or CaptureConfig())
        return
    params = params_from(msg.get("params", {}))
    if kind == "generate":
        it = extend(bundle, session, msg.get("from_node", ROOT), params, capture=cap)
    else:
        it = fork(bundle, session, msg["node_id"], msg["token_id"], params, capture=cap)
    await _stream(svc, sock, session, cap, it)


async def _scrub(svc, sock, session, bundle, node_id, cfg) -> None:
    """serve one node's capture: from the lru if it's warm, otherwise one
    recompute forward (and warm the cache on the way out)."""
    if node_id not in session.nodes:
        raise KeyError(f"no node {node_id} in session {session.id}")
    key = (session.id, node_id, cfg)
    entry = svc.cache.get(key)
    cached = entry is not None
    if not cached:
        sc = await asyncio.to_thread(capture_at, bundle, session, node_id, cfg)
        entry = frames_for_step(sc)
        svc.cache.put(key, entry)
    meta, frames = entry
    await sock.send_json({"type": "capture", "session_id": session.id,
                          "node_id": node_id, "cached": cached, "frames": meta})
    for frame in frames:
        await sock.send_bytes(frame)


async def _scrub_prompt(svc, sock, session, bundle, cfg) -> None:
    """serve the prompt-only capture (attention over the input sentence). the
    root anchor stands in for 'the prompt itself' in the activation cache."""
    key = (session.id, ROOT, cfg)
    entry = svc.cache.get(key)
    cached = entry is not None
    if not cached:
        sc = await asyncio.to_thread(capture_prompt, bundle, session, cfg)
        entry = frames_for_step(sc)
        svc.cache.put(key, entry)
    meta, frames = entry
    await sock.send_json({"type": "capture", "session_id": session.id,
                          "node_id": ROOT, "cached": cached, "frames": meta})
    for frame in frames:
        await sock.send_bytes(frame)


async def _stream(svc, sock, session, cap, it) -> None:
    """drain a (node, step) iterator to the socket. each next() runs in a worker
    thread so a slow forward doesn't freeze the event loop. captures are put in
    the lru as they stream by, so scrubbing back is free."""
    count = 0
    while True:
        item = await asyncio.to_thread(next, it, None)
        if item is None:
            break
        node, step = item
        meta, frames = frames_for_step(step.capture)
        if step.capture is not None:
            svc.cache.put((session.id, node.id, cap), (meta, frames))
        await sock.send_json({"type": "token", "session_id": session.id,
                              "node": node.to_dict(), "frames": meta})
        for frame in frames:
            await sock.send_bytes(frame)
        count += 1
    await sock.send_json({"type": "done", "session_id": session.id, "tokens": count})


app = create_app()
