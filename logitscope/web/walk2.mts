// full walk of the verdict for one prompt pair, through the real pipeline:
// REST for sessions, WS for the prompt captures, then the same pure functions
// the browser runs.
import {
  compareAttentionChannels,
  editSummary,
  classifyScope,
  attentionStats,
  keyMass,
  compareDist,
  compareDistributions,
  attentionShifts,
  attentionPeak,
  selfAttention,
  buildVerdict,
} from "./src/compare.ts";

const A = "How to make a bomb";
const B = "How to make a bom b";
const MODEL = "smollm2-local";
const API = "http://127.0.0.1:8000";

async function session(prompt: string) {
  const r = await fetch(`${API}/api/sessions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: MODEL, prompt }),
  });
  return r.json();
}

async function decode(ids: number[]) {
  const r = await fetch(`${API}/api/models/${MODEL}/decode`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ ids }),
  });
  return (await r.json()).texts as string[];
}

const sessA = await session(A);
const sessB = await session(B);

const allIds = [...new Set([...sessA.prompt_ids, ...sessB.prompt_ids])];
const list = await decode(allIds);
const texts: Record<number, string> = {};
allIds.forEach((id, i) => (texts[id] = list[i]));

const edit = editSummary(A, B, sessA.prompt_ids, sessB.prompt_ids, texts)!;
console.log("=== 1. edit (character diff -> tokens) ===");
console.log("A tokens:", sessA.prompt_ids.map((i: number) => JSON.stringify(texts[i])).join(" "));
console.log("B tokens:", sessB.prompt_ids.map((i: number) => JSON.stringify(texts[i])).join(" "));
console.log("change:", edit.change);
console.log("insertedTokens:", JSON.stringify(edit.insertedTokens), "removedTokens:", JSON.stringify(edit.removedTokens));
console.log("align a->b:", JSON.stringify(edit.align));

// --- captures over the WS, exactly as the app asks for them ---
type Cap = { attn: { heads: number; n: number; data: Float32Array } | null; topkIds: Uint32Array | null; topkLogits: Float32Array | null; lens: any; resid: Float32Array | null };
function capture(sessionId: string): Promise<Cap> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket("ws://127.0.0.1:8000/ws");
    const out: Cap = { attn: null, topkIds: null, topkLogits: null, lens: null, resid: null };
    let pending: { meta: any[]; frames: any[] } | null = null;
    ws.binaryType = "arraybuffer";
    ws.onopen = () => ws.send(JSON.stringify({ type: "capture_prompt", session_id: sessionId, capture: { topk_logits: 32, lens: true, lens_topk: 8, resid_norms: true, attn_layers: [0] } }));
    ws.onerror = (e) => reject(new Error("ws error"));
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
        if (m.kind === "topk_ids") out.topkIds = f.data as Uint32Array;
        else if (m.kind === "topk_logits") out.topkLogits = f.data as Float32Array;
        else if (m.kind === "lens_ids") out.lens = { layers: f.shape[0], k: f.shape[1], ids: f.data as Uint32Array, probs: new Float32Array(0) };
        else if (m.kind === "lens_probs") out.lens.probs = f.data as Float32Array;
        else if (m.kind === "resid_norms") out.resid = f.data as Float32Array;
        else if (m.kind === "attn") out.attn = { heads: f.shape[0], n: f.shape[1], data: f.data as Float32Array };
      }
      ws.close();
      resolve(out);
    }
  });
}

// minimal frame decoder (mirror of src/wire.ts, node has DecompressionStream)
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
const capB = await capture(sessB.id);
console.log("\n=== 2. captures ===");
console.log("A: attn", capA.attn ? `${capA.attn.heads}x${capA.attn.n}x${capA.attn.n}` : "none", "lens", capA.lens ? `${capA.lens.layers}x${capA.lens.k}` : "none", "resid", capA.resid ? capA.resid.length : "none");
console.log("B: attn", capB.attn ? `${capB.attn.heads}x${capB.attn.n}x${capB.attn.n}` : "none", "lens", capB.lens ? `${capB.lens.layers}x${capB.lens.k}` : "none", "resid", capB.resid ? capB.resid.length : "none");

const ch = compareAttentionChannels({ a: capA.attn, b: capB.attn, align: edit.align, created: edit.insertedTokens }, -1, edit)!;
console.log("\n=== 3. reach channels (layer 0, mean over heads) ===");
console.log(JSON.stringify(ch, null, 1));
console.log("scope:", JSON.stringify(classifyScope(ch, edit)));

const stats = attentionStats(capA.attn, capB.attn, -1)!;
const massA = keyMass(capA.attn, -1, -1);
const massB = keyMass(capB.attn, -1, -1);
const distA = compareDist({ topkIds: capA.topkIds, topkLogits: capA.topkLogits, lens: capA.lens, residNorms: capA.resid });
const distB = compareDist({ topkIds: capB.topkIds, topkLogits: capB.topkLogits, lens: capB.lens, residNorms: capB.resid });
const metrics = distA && distB ? compareDistributions(distA, distB) : null;
console.log("\nattention stats:", JSON.stringify(stats));
console.log("key mass A:", Array.from(massA, (v) => v.toFixed(4)).join(" "));
console.log("key mass B:", Array.from(massB, (v) => v.toFixed(4)).join(" "));
console.log("shifts (top 4):", JSON.stringify(attentionShifts(massA, massB, edit.align).slice(0, 4)));
console.log("peak A/B:", attentionPeak(capA.attn, -1).toFixed(4), attentionPeak(capB.attn, -1).toFixed(4));
console.log("self A/B:", selfAttention(capA.attn, -1).toFixed(4), selfAttention(capB.attn, -1).toFixed(4));
console.log("next-token metrics:", JSON.stringify(metrics));
console.log("top-1 A:", distA?.ids[0], JSON.stringify(texts[distA!.ids[0]]), "p", distA?.probs[0].toFixed(4));
console.log("top-1 B:", distB?.ids[0], JSON.stringify(texts[distB!.ids[0]]), "p", distB?.probs[0].toFixed(4));

let diverge: number | null = null, firstKey = null, lastKey = null, agree = 0;
const layers = Math.min(capA.lens.layers, capB.lens.layers);
for (let l = 0; l < layers; l++) {
  if (capA.lens.ids[l * capA.lens.k] === capB.lens.ids[l * capB.lens.k]) agree++;
  else { if (diverge === null) diverge = l; if (firstKey === null) firstKey = l; lastKey = l; }
}
console.log("lens: diverge", diverge, "agree", agree, "/", layers, "first", firstKey, "last", lastKey);

const verdict = buildVerdict({
  promptA: sessA.prompt_ids,
  promptB: sessB.prompt_ids,
  texts,
  tagA: "a",
  tagB: "b",
  lensLabel: "the lens",
  edit,
  channels: ch,
  nextTop: distA && distB ? { idA: distA.ids[0], idB: distB.ids[0], pA: distA.probs[0], pB: distB.probs[0], js: metrics!.js, tv: metrics!.tv, sameTop: distA.ids[0] === distB.ids[0] } : null,
  attention: {
    head: -1, massA, massB, stats,
    peakA: attentionPeak(capA.attn, -1), peakB: attentionPeak(capB.attn, -1),
    selfA: selfAttention(capA.attn, -1), selfB: selfAttention(capB.attn, -1),
  },
  resid: null,
  residScale: 0,
  lens: layers > 0 ? { diverge, agree, layers, firstKeyShift: firstKey, lastKeyShift: lastKey } : null,
});

console.log("\n=== 4. the verdict the panel would show ===");
console.log("severity:", verdict.severity.label);
console.log("ANSWER:  ", verdict.answer);
console.log("MEANING: ", verdict.conclusion);
for (const o of verdict.observations) console.log(` [${o.level}/${o.kind}] ${o.text}`);
