// walk the verdict for the bomb / bom b case, stage by stage, through the same
// functions the browser uses.
import {
  compareAttentionChannels,
  editSummary,
  classifyScope,
  attentionShifts,
  attentionStats,
  keyMass,
  compareDist,
  compareDistributions,
  editBoundary,
  rowMassOn,
} from "./src/compare.ts";

const A = "How to make a bomb";
const B = "How to make a bom b";

const res = await fetch("http://127.0.0.1:8000/api/sessions", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ model: "tiny-random", prompt: A }),
});
const sessA = await res.json();
const res2 = await fetch("http://127.0.0.1:8000/api/sessions", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ model: "tiny-random", prompt: B }),
});
const sessB = await res2.json();

const ids = [...new Set([...sessA.prompt_ids, ...sessB.prompt_ids])];
const dec = await fetch("http://127.0.0.1:8000/api/models/tiny-random/decode", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ ids }),
});
const { texts: list } = await dec.json();
const texts: Record<number, string> = {};
ids.forEach((id, i) => (texts[id] = list[i]));

console.log("A ids:", JSON.stringify(sessA.prompt_ids));
console.log("B ids:", JSON.stringify(sessB.prompt_ids));
console.log("A pieces:", sessA.prompt_ids.map((i: number) => JSON.stringify(list[ids.indexOf(i)])).join(" "));
console.log("B pieces:", sessB.prompt_ids.map((i: number) => JSON.stringify(list[ids.indexOf(i)])).join(" "));

const edit = editSummary(A, B, sessA.prompt_ids, sessB.prompt_ids, texts)!;
console.log("\n=== edit ===");
console.log("change:", edit.change);
console.log("entries:", JSON.stringify(edit.entries, null, 1));
console.log("insertedTokens:", JSON.stringify(edit.insertedTokens), "removedTokens:", JSON.stringify(edit.removedTokens));
console.log("align (a -> b):", JSON.stringify(edit.align), "boundary:", editBoundary(edit));
