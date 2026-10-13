// throwaway: does the server actually return a different square per attn layer?
const API = "http://127.0.0.1:8000";
const MODEL = "smollm2-local";

async function post(path: string, body: unknown) {
  const r = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json();
}

async function decodeFrame(buf: ArrayBuffer) {
  const v = new DataView(buf);
  const kind = v.getUint8(3);
  const dtype = v.getUint8(4);
  const ndim = v.getUint8(5);
  let off = 6;
  const shape: number[] = [];
  for (let i = 0; i < ndim; i++) { shape.push(v.getUint32(off, true)); off += 4; }
  const raw = new Uint8Array(buf, off);
  const bytes = new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());
  let data: any;
  if (dtype === 1) {
    const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
    data = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      const b = u16[i];
      const s = b & 0x8000 ? -1 : 1, e = (b >> 10) & 0x1f, f = b & 0x3ff;
      data[i] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
    }
  } else if (dtype === 2) data = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  else data = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  return { kind, shape, data };
}

function capturePrompt(sessionId: string, layer: number): Promise<any> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket("ws://127.0.0.1:8000/ws");
    let pending: any = null;
    let attn: any = null;
    let metaLayer = -1;
    ws.binaryType = "arraybuffer";
    ws.onopen = () => ws.send(JSON.stringify({ type: "capture_prompt", session_id: sessionId, capture: { topk_logits: 4, attn_layers: [layer] } }));
    ws.onerror = () => reject(new Error("ws error"));
    ws.onmessage = async (ev) => {
      if (typeof ev.data === "string") {
        const msg = JSON.parse(ev.data);
        if (msg.type === "error") return reject(new Error(msg.error));
        if (msg.type === "capture") {
          pending = { meta: msg.frames, frames: [] };
          if (msg.frames.length === 0) finish();
        }
        return;
      }
      pending.frames.push(await decodeFrame(ev.data as ArrayBuffer));
      if (pending.frames.length === pending.meta.length) finish();
    };
    function finish() {
      for (let i = 0; i < pending.meta.length; i++) {
        const m = pending.meta[i];
        if (m.kind === "attn") {
          metaLayer = m.layer;
          attn = pending.frames[i];
        }
      }
      ws.close();
      resolve({ metaLayer, n: attn ? attn.shape[1] : 0, sample: attn ? Array.from(attn.data.slice(0, 8), (x: number) => x.toFixed(5)).join(" ") : "" });
    }
  });
}

const A = await post("/api/sessions", { model: MODEL, prompt: "How to make a bomb" });
const B = await post("/api/sessions", { model: MODEL, prompt: "How to make a bom b" });

for (const layer of [0, 1, 3, 20, 29]) {
  const ra = await capturePrompt(A.id, layer);
  const rb = await capturePrompt(B.id, layer);
  console.log(`layer ${layer}: server says frame.layer=${ra.metaLayer} | A n=${ra.n} [${ra.sample}] | B n=${rb.n} [${rb.sample}]`);
}
