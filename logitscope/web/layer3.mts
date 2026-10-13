// throwaway: recompute layer-3 channels with both sides freshly captured,
// to find out whether the discrepancy is in the capture or in the comparison.
import { compareAttentionChannels, editSummary, keyMass, attentionShifts } from "./src/compare.ts";

const API = "http://127.0.0.1:8000";
const MODEL = "smollm2-local";
const A = "How to make a bomb";
const B = "How to make a bom b";

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
  const kind = v.getUint8(3), dtype = v.getUint8(4), ndim = v.getUint8(5);
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
      const b = u16[i]; const s = b & 0x8000 ? -1 : 1, e = (b >> 10) & 0x1f, f = b & 0x3ff;
      data[i] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
    }
  } else if (dtype === 2) data = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  else data = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  return { kind, shape, data };
}
function cap(sessionId: string, layer: number, extras = false): Promise<any> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket("ws://127.0.0.1:8000/ws");
    let pending: any = null;
    const out: any = { attn: null };
    ws.binaryType = "arraybuffer";
    ws.onopen = () =>
      ws.send(JSON.stringify({ type: "capture_prompt", session_id: sessionId, capture: extras ? { topk_logits: 32, lens: true, lens_topk: 8, resid_norms: true, attn_layers: [layer] } : { topk_logits: 4, attn_layers: [layer] } }));
    ws.onerror = () => reject(new Error("ws error"));
    ws.onmessage = async (ev) => {
      if (typeof ev.data === "string") {
        const msg = JSON.parse(ev.data);
        if (msg.type === "error") return reject(new Error(msg.error));
        if (msg.type === "capture") { pending = { meta: msg.frames, frames: [] }; if (msg.frames.length === 0) finish(); }
        return;
      }
      pending.frames.push(await decodeFrame(ev.data as ArrayBuffer));
      if (pending.frames.length === pending.meta.length) finish();
    };
    function finish() {
      for (let i = 0; i < pending.meta.length; i++) {
        const m = pending.meta[i], f = pending.frames[i];
        if (m.kind === "attn") out.attn = { heads: f.shape[0], n: f.shape[1], data: f.data, layer: m.layer };
      }
      ws.close();
      resolve(out);
    }
  });
}

const sa = await post("/api/sessions", { model: MODEL, prompt: A });
const sb = await post("/api/sessions", { model: MODEL, prompt: B });
const ids = [...new Set([...sa.prompt_ids, ...sb.prompt_ids])];
const { texts: list } = await post(`/api/models/${MODEL}/decode`, { ids });
const texts: Record<number, string> = {};
ids.forEach((id, i) => (texts[id] = list[i]));
const edit = editSummary(A, B, sa.prompt_ids, sb.prompt_ids, texts)!;

console.log("\n=== corrected scan: both sides at the same layer ===");
console.log("layer | redist% | rows | nonLocalRows | nonLocal% | scope");
for (let layer = 0; layer < 30; layer++) {
  const ca = await cap(sa.id, layer);
  const cb = await cap(sb.id, layer);
  const ch = compareAttentionChannels({ a: ca.attn, b: cb.attn, align: edit.align, created: edit.insertedTokens }, -1, edit);
  if (!ch) { console.log(`${String(layer).padStart(5)} | no comparable keys`); continue; }
  const kind = ch.nonLocalRows >= Math.max(2, Math.ceil(ch.rowsTotal * 0.1)) && ch.nonLocalShare >= 0.4 ? "global" : ch.nonLocalRows <= 1 && ch.nonLocalShare < 0.25 ? "local" : "mixed";
  console.log(
    `${String(layer).padStart(5)} | ${(ch.redistributedMass * 100).toFixed(2).padStart(6)}% | ${String(ch.rowsChanged).padStart(4)} | ${String(ch.nonLocalRows).padStart(12)} | ${(ch.nonLocalShare * 100).toFixed(1).padStart(8)}% | ${kind}`,
  );
}
