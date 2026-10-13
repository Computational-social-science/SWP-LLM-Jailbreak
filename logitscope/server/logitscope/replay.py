"""replay mode: the server performs a recording instead of running a model.

recordings come from scripts/record_replay.py. this app keeps exactly the
rest surface the ui touches and answers ws requests with the recorded byte
streams — deterministic, instant, and it never imports torch, so the
playwright e2e (and anyone else's ci) runs without downloading a single
weight.

    python -m logitscope --replay tests/fixtures/replay.jsonl
"""

import asyncio
import base64
import json
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect

# (kind, payload): kind is "text" or "bytes"
WireItem = tuple[str, str | bytes]


class Recording:
    def __init__(self, path: str | Path):
        self.models: list[dict] = []
        self.session: dict | None = None
        self.session_final: dict | None = None
        self.decode: dict[int, str] = {}
        self.groups: dict[str, list[WireItem]] = {}
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            kind = rec["kind"]
            if kind == "meta":
                pass  # documentation for humans, not for the player
            elif kind == "models":
                self.models = rec["data"]
            elif kind == "session":
                self.session = rec["data"]
            elif kind == "session_final":
                self.session_final = rec["data"]
            elif kind == "decode":
                self.decode = {int(i): t for i, t in rec["map"].items()}
            elif kind == "ws":
                item: WireItem = (
                    ("text", rec["text"])
                    if "text" in rec
                    else ("bytes", base64.b64decode(rec["frame_b64"]))
                )
                self.groups.setdefault(rec["group"], []).append(item)
            else:
                raise ValueError(f"unknown record kind {kind!r}")
        if self.session is None or not self.groups:
            raise ValueError(f"{path} doesn't look like a recording")


def capture_key(msg: dict) -> str:
    """same loose key the recorder writes: node + first attn layer. the real
    server keys its lru on the whole capture config; a recording only ever
    holds one non-attn config per node, so this is enough to find it."""
    cap = msg.get("capture") or {}
    layers = cap.get("attn_layers")
    return f"capture:{msg.get('node_id')}:{layers[0] if layers else 'none'}"


def create_replay_app(path: str | Path) -> FastAPI:
    rec = Recording(path)
    app = FastAPI(title="logitscope-replay")

    @app.get("/api/health")
    def health():
        return {"ok": True, "replay": True}

    @app.get("/api/models")
    def models():
        return rec.models

    @app.post("/api/models/load")
    def load_model():
        return {"loaded": True}  # everything is already "loaded" in a recording

    @app.get("/api/models/progress")
    def progress():
        return {}

    @app.post("/api/models/{name}/decode")
    async def decode(name: str, request: Request):
        body = await request.json()
        return {"texts": [rec.decode.get(int(i), "?") for i in body["ids"]]}

    @app.post("/api/sessions")
    def create_session():
        # whatever prompt arrives, the recording is the session you get
        return rec.session

    @app.get("/api/sessions")
    def list_sessions():
        s = rec.session_final or rec.session
        return [{"id": s["id"], "model": s["model"], "prompt": s["prompt"],
                 "nodes": len(s["nodes"]), "created": s["created"]}]

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str):
        return rec.session_final or rec.session

    @app.get("/api/cache/stats")
    def cache_stats():
        return {"entries": 0, "bytes": 0, "max_bytes": 0,
                "hits": 0, "misses": 0, "evictions": 0}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            while True:
                msg = await sock.receive_json()
                kind = msg.get("type")
                key = capture_key(msg) if kind == "capture" else kind
                items = rec.groups.get(key)
                if not items:
                    await sock.send_json(
                        {"type": "error", "error": f"nothing recorded for {key!r}"})
                    continue
                for item_kind, payload in items:
                    if item_kind == "text":
                        await sock.send_text(payload)  # type: ignore[arg-type]
                    else:
                        await sock.send_bytes(payload)  # type: ignore[arg-type]
                    # a hint of streaming feel instead of one instant blast
                    await asyncio.sleep(0.01)
        except WebSocketDisconnect:
            pass

    return app
