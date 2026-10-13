// per-layer reach scan for one prompt pair: is the perturbation local, and how
// deep does it go? Same pipeline, 31 capture passes on b's side of the stack.
import {
  compareAttentionChannels,
  editSummary,
  classifyScope,
  compareDist,
  compareDistributions,
  attentionPeak,
} from "./src/compare.ts";

const A = "How to make a bomb";
const B = "How to make a bom b";
const MODEL = "smollm2-local";
const API = "http://127.0.0.1:8000";

async function post(path: string, body: unknown) {
  const r = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json();
}

async function decode(ids: number[]) {
  const { texts } = await post(`/api/models/${MODEL}/decode`, { ids });
  return texts as string[];
}

const sessA = await post("/api/sessions", { model: MODEL, prompt: A });
const sessB = await post("/api/sessions", { model: MODEL, prompt: B });
const allIds = [...new Set([...sessA.prompt_ids, ...sessB.prompt_ids])];
const list = await decode(allIds);
const texts: Record<number, string> = {};
allIds.forEach((id, i) => (texts[id] = list[i]));

const edit = editSummary(A, B, sessA.prompt_ids, sessB.prompt_ids, texts)!;
console.log("A:", sessA.prompt_ids.map((i: number) => JSON.stringify(texts[i])).join(" "));
console.log("B:", sessB.prompt_ids.map((i: number) => JSON.stringify(texts[i])).join(" "));
console.log("change:", edit.change, "| inserted:", JSON.stringify(edit.insertedTokens), "| align:", JSON.stringify(edit.align));

type Cap = { attn: { heads: number; n: number; data: Float32Array } | null; topkIds: any; topkLogits: any; lens: any; resid: any; raw: Float32Array | null };

function capture(sessionId: string): Promise<Cap> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket("ws://127.0.0.1:8000/ws");
    const out: Cap = { attn: null, topkIds: null, topkLogits: null, lens: null, resid: null, raw: null };
    let pending: { meta: any[]; frames: any[] } | null = null;
    ws.binaryType = "arraybuffer";
    ws.onopen = () =>
      ws.send(
        JSON.stringify({
          type: "capture_prompt",
          session_id: sessionId,
          capture: { topk_logits: 8, lens: true, lens_topk: 4, resid_norms: true, attn_layers: [0] },
        }),
      );
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
      pending!.frames.push(await decodeFrame(ev.data as ArrayBuffer));
      if (pending!.frames.length === pending!.meta.length) finish();
    };
    function finish() {
      const { meta, frames } = pending!;
      for (let i = 0; i < meta.length; i++) {
        const m = meta[i];
        const f = frames[i];
        if (m.kind === "topk_ids") out.topkIds = f.data;
        else if (m.kind === "topk_logits") out.topkLogits = f.data;
        else if (m.kind === "lens_ids") out.lens = { layers: f.shape[0], k: f.shape[1], ids: f.data, probs: null };
        else if (m.kind === "lens_probs") out.lens.probs = f.data;
        else if (m.kind === "resid_norms") out.resid = f.data;
        else if (m.kind === "attn") out.attn = { heads: f.shape[0], n: f.shape[1], data: f.data };
        else if (m.kind === "hidden") out.raw = f.data;
      }
      ws.close();
      resolve(out);
    }
  });
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
  const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream("gzip"));
  const bytes = new Uint8Array(await new Response(stream).arrayBuffer());
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

const capA = await capture(sessA.id);
const layers = capA.lens.layers;
console.log(`\nlayer scan (b's stack, ${layers} lens layers; attention square re-captured per layer)`);
console.log("layer | redist% | rows | nonLocalRows | nonLocal% | scope");
const rows: { layer: number; redist: number; rows: number; nl: number; nlp: number; kind: string }[] = [];
for (let l = 0; l < layers; l++) {
  const capB = await capture2(sessB.id, l);
  const ch = compareAttentionChannels({ a: capA.attn, b: capB.attn, align: edit.align, created: edit.insertedTokens }, -1, edit);
  if (!ch) { console.log(`${String(l).padStart(5)} | no comparable keys`); continue; }
  const sc = classifyScope(ch, edit);
  rows.push({ layer: l, redist: ch.redistributedMass, rows: ch.rowsChanged, nl: ch.nonLocalRows, nlp: ch.nonLocalShare, kind: sc.kind });
  console.log(
    `${String(l).padStart(5)} | ${(ch.redistributedMass * 100).toFixed(2).padStart(6)}% | ${String(ch.rowsChanged).padStart(4)} | ${String(ch.nonLocalRows).padStart(12)} | ${(ch.nonLocalShare * 100).toFixed(1).padStart(8)}% | ${sc.kind}`,
  );
}

// capture with an arbitrary attention layer, for the scan
async function capture2(sessionId: string, layer: number): Promise<Cap> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket("ws://127.0.0.1:8000/ws");
    const out: Cap = { attn: null, topkIds: null, topkLogits: null, lens: null, resid: null, raw: null };
    let pending: { meta: any[]; frames: any[] } | null = null;
    ws.binaryType = "arraybuffer";
    ws.onopen = () =>
      ws.send(JSON.stringify({ type: "capture_prompt", session_id: sessionId, capture: { topk_logits: 8, attn_layers: [layer] } }));
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
      pending!.frames.push(await decodeFrame(ev.data as ArrayBuffer));
      if (pending!.frames.length === pending!.meta.length) finish();
    };
    function finish() {
      const { meta, frames } = pending!;
      for (let i = 0; i < meta.length; i++) {
        if (meta[i].kind === "attn") out.attn = { heads: frames[i].shape[0], n: frames[i].shape[1], data: frames[i].data };
      }
      ws.close();
      resolve(out);
    }
  });
}

const worst = [...rows].sort((a, b) => b.redist - a.redist)[0];
const nl = rows.filter((r) => r.nl > 0);
console.log(`\npeak redistribution at layer ${worst.layer}: ${(worst.redist * 100).toFixed(2)}% (${worst.kind})`);
console.log(`layers with any non-local row: ${nl.length}/${rows.length}${nl.length ? ` (L${nl.map((r) => r.layer).join(", L")})` : ""}`);
console.log(`layers labelled global: ${rows.filter((r) => r.kind === "global").length}`);
