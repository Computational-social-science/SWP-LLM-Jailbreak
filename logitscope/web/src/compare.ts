// numeric analysis behind the two-column compare view: what changed between
// two input sentences. pure functions, no react, no canvas — the component
// renders them and the tests pin them down.
//
// the two sides are the same model twice, so everything here is a *paired*
// comparison: character-level diff of the raw text (whitespace is the point of
// this view, and the tokenizer may hide it), and per-cell / per-layer deltas of
// the two capture passes.

import { entropyNats } from "./math";
import { visible, visibleForDiff } from "./tokens";

// -- text diff --

/** edit ops between two strings. a = chars only in a, b = chars only in b,
 * same = an aligned char pair. one dp; prompts are short. */
export type TextOp =
  | { kind: "same"; a: number; b: number }
  | { kind: "del"; a: number }
  | { kind: "ins"; b: number };

export function diffStrings(a: string, b: string, maxCells = 4_000_000): TextOp[] {
  const n = a.length;
  const m = b.length;
  // a whole-paragraph prompt is not worth an O(n*m) table in the ui: past the
  // budget, report the change as one wholesale replacement
  if (n * m > maxCells) {
    const out: TextOp[] = [];
    for (let i = 0; i < n; i++) out.push({ kind: "del", a: i });
    for (let j = 0; j < m; j++) out.push({ kind: "ins", b: j });
    return out;
  }

  const dp: Uint32Array[] = [];
  for (let i = 0; i <= n; i++) dp.push(new Uint32Array(m + 1));
  for (let i = 1; i <= n; i++)
    for (let j = 1; j <= m; j++)
      dp[i][j] =
        a[i - 1] === b[j - 1] ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);

  const ops: TextOp[] = [];
  let i = n;
  let j = m;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && a[i - 1] === b[j - 1]) {
      ops.push({ kind: "same", a: i - 1, b: j - 1 });
      i--;
      j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      ops.push({ kind: "ins", b: j - 1 });
      j--;
    } else {
      ops.push({ kind: "del", a: i - 1 });
      i--;
    }
  }
  ops.reverse();
  // granular on purpose: diffMarks reads positions, and a run of ops is what
  // tells it where a replacement starts and ends
  return ops;
}

export interface DiffMarks {
  /** index -> true when that char differs from its aligned counterpart */
  changedA: boolean[];
  changedB: boolean[];
  /** input index -> aligned index on the other side, or -1 when the char is
   * fresh (inserted on b, deleted on a). -1 rather than null because a sparse
   * array hole reads back as undefined, which is not a value to trust. */
  alignA: number[];
  alignB: number[];
  /** changed index runs, for a compact textual summary */
  delRuns: [number, number][];
  insRuns: [number, number][];
}

/** the two rows of a paired character diff. an inserted run that sits right
 * after a deleted run is a replacement: the pair gets aligned, so the ui can
 * draw the changed char red on one side and phosphor on the other. */
export function diffMarks(a: string, b: string): DiffMarks {
  const ops = diffStrings(a, b);
  const changedA: boolean[] = new Array(a.length).fill(false);
  const changedB: boolean[] = new Array(b.length).fill(false);
  const alignA: number[] = new Array(a.length).fill(-1);
  const alignB: number[] = new Array(b.length).fill(-1);
  const delRuns: [number, number][] = [];
  const insRuns: [number, number][] = [];

  // the pending runs, settled by flush(). both are [lo, hi] and empty until
  // an op of that kind is seen — never sentinel counters, or "no insert run"
  // would size out as length 1 and pair a delete with nothing.
  let del: [number, number] | null = null;
  let ins: [number, number] | null = null;

  const flush = (): void => {
    if (ins) {
      const [insLo, insHi] = ins;
      for (let p = 0; insLo + p <= insHi; p++) {
        changedB[insLo + p] = true;
        // the head of the insert run pairs with the delete run before it
        if (del && p <= del[1] - del[0]) alignB[insLo + p] = del[0] + p;
      }
      insRuns.push([insLo, insHi]);
    }
    if (del) {
      const [delLo, delHi] = del;
      for (let p = 0; delLo + p <= delHi; p++) {
        changedA[delLo + p] = true;
        // only the head of a delete run stands in for an inserted char
        if (ins && p <= ins[1] - ins[0]) alignA[delLo + p] = ins[0] + p;
      }
      delRuns.push([delLo, delHi]);
    }
    del = null;
    ins = null;
  };

  for (const op of ops) {
    if (op.kind === "same") {
      flush();
      alignA[op.a] = op.b;
      alignB[op.b] = op.a;
    } else if (op.kind === "del") {
      // a new delete run starts only when the previous one was already
      // settled: op order is same -> del* -> ins* -> same -> ...
      if (ins) flush();
      del = del ? [del[0], op.a] : [op.a, op.a];
    } else {
      ins = ins ? [ins[0], op.b] : [op.b, op.b];
    }
  }
  flush();

  return { changedA, changedB, alignA, alignB, delRuns, insRuns };
}

// -- next-token distribution --

export interface CompareSideCapture {
  topkIds: Uint32Array | null;
  topkLogits: Float32Array | null;
  lens: { layers: number; k: number; ids: Uint32Array; probs: Float32Array } | null;
  residNorms: Float32Array | null;
}

export interface CompareDist {
  /** top-k ids, most likely first */
  ids: number[];
  /** exact softmax over the top-k logits */
  probs: Float32Array;
  /** the raw top-k logits. kept because the probabilities can underflow in f32
   * for a long tail, and the least likely candidate is exactly where the two
   * columns tend to differ most */
  logits: Float32Array;
  logZ: number;
  /** entropy of the top-k mass, in nats. the tail of the vocab is not in the
   * frame, so this is a lower bound on the true entropy — every column pays
   * the same one, which is what makes the two comparable */
  entropy: number;
  /** the top of the last lens layer — the model's own final answer */
  lensTop: { id: number; p: number } | null;
}

/** p(id) for a distribution, recomputed in log space so a tail candidate stays
 * a small probability instead of a zero. a missing side is probability 0. */
export function probIn(dist: CompareDist | null, id: number): number {
  if (!dist) return 0;
  const i = dist.ids.indexOf(id);
  return i < 0 ? 0 : Math.exp(dist.logits[i] - dist.logZ);
}

export function compareDist(side: CompareSideCapture): CompareDist | null {
  const ids = side.topkIds;
  const logits = side.topkLogits;
  if (!ids || !logits || ids.length === 0) return null;
  // log-sum-exp first: exp(logit - logZ) is stable where a naive softmax
  // divides small numbers by a large one and flushes the tail to zero
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - max);
  const logZ = max + Math.log(sum);
  const probs = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) probs[i] = Math.exp(logits[i] - logZ);

  const lens = side.lens;
  // lens probabilities come from the server's own softmax; recompute the last
  // row here so a lens chip and a candidate bar never disagree on the same token
  const lensTop =
    lens && lens.k > 0 && lens.layers > 0
      ? { id: lens.ids[(lens.layers - 1) * lens.k], p: lens.probs[(lens.layers - 1) * lens.k] }
      : null;
  return {
    ids: Array.from(ids),
    probs,
    logits: Float32Array.from(logits),
    logZ,
    entropy: entropyNats(probs),
    lensTop,
  };
}

export interface DistMetrics {
  /** total variation: 0.5 * sum |p - q| — 1 means disjoint supports */
  tv: number;
  /** KL(a || b) over the shared top-k ids, in nats (null when q has a zero) */
  kl: number | null;
  /** symmetric and always finite: the right headline number */
  js: number;
  /** where the two distributions disagree most */
  maxDelta: { id: number; delta: number } | null;
  sameTop: boolean;
}

/** merge both sides' candidates, ordered by their best rank on either side —
 * the tokens worth drawing a paired bar for. */
export function unionCandidates(a: CompareDist, b: CompareDist, k: number): number[] {
  const rank = new Map<number, number>();
  a.ids.forEach((id, i) => rank.set(id, Math.min(rank.get(id) ?? Infinity, i)));
  b.ids.forEach((id, i) => rank.set(id, Math.min(rank.get(id) ?? Infinity, i)));
  return [...rank.entries()]
    .sort((x, y) => x[1] - y[1])
    .slice(0, k)
    .map(([id]) => id);
}

/** id -> probability over the union of both sides' candidates. the values come
 * from probIn, so a candidate whose f32 probability underflowed still carries
 * the small number it deserves instead of a hard zero. */
function distMap(d: CompareDist | null): Map<number, number> {
  const out = new Map<number, number>();
  if (!d) return out;
  for (const id of d.ids) out.set(id, probIn(d, id));
  return out;
}

export function compareDistributions(a: CompareDist, b: CompareDist): DistMetrics {
  const pa = distMap(a);
  const pb = distMap(b);
  const keys = new Set([...pa.keys(), ...pb.keys()]);

  let tv = 0;
  let maxDelta: { id: number; delta: number } | null = null;
  for (const id of keys) {
    const d = (pa.get(id) ?? 0) - (pb.get(id) ?? 0);
    tv += Math.abs(d);
    if (!maxDelta || Math.abs(d) > Math.abs(maxDelta.delta)) maxDelta = { id, delta: d };
  }
  tv /= 2;

  // KL over the shared support only: an honest "how far did it move", but a
  // token with mass on one side and none on the other leaves it undefined
  let kl: number | null = 0;
  for (const [id, p] of pa) {
    if (p <= 0) continue;
    const q = pb.get(id) ?? 0;
    if (q <= 0) {
      kl = null;
      break;
    }
    kl += p * Math.log(p / q);
  }

  const m = new Map<number, number>();
  for (const id of keys) m.set(id, 0.5 * ((pa.get(id) ?? 0) + (pb.get(id) ?? 0)));
  const js = 0.5 * klTo(pa, m) + 0.5 * klTo(pb, m);

  return { tv, kl, js, maxDelta, sameTop: a.ids[0] === b.ids[0] };
}

function klTo(p: Map<number, number>, q: Map<number, number>): number {
  let sum = 0;
  for (const [id, pv] of p) {
    if (pv <= 0) continue;
    const qv = q.get(id) ?? 0;
    if (qv <= 0) return 0; // outside the support: contributes nothing to js
    sum += pv * Math.log(pv / qv);
  }
  return sum;
}

// -- attention --

export interface AttentionStats {
  /** biggest single-cell change, and where it happened */
  maxAbs: number;
  at: { i: number; j: number } | null;
  meanAbs: number;
  /** largest weight in each matrix, for scale */
  maxA: number;
  maxB: number;
}

export function attentionStats(a: AttentionMatrix, b: AttentionMatrix, head: number): AttentionStats | null {
  if (a.n !== b.n || a.heads !== b.heads) return null;
  const n = a.n;
  let maxAbs = 0;
  let at: { i: number; j: number } | null = null;
  let sumAbs = 0;
  let count = 0;
  let maxA = 0;
  let maxB = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      const wa = headAt(a.data, a.heads, n, head, i, j);
      const wb = headAt(b.data, b.heads, n, head, i, j);
      const d = Math.abs(wa - wb);
      if (d > maxAbs) {
        maxAbs = d;
        at = { i, j };
      }
      if (wa > maxA) maxA = wa;
      if (wb > maxB) maxB = wb;
      sumAbs += d;
      count++;
    }
  }
  return { maxAbs, at, meanAbs: count > 0 ? sumAbs / count : 0, maxA, maxB };
}

export interface AttentionMatrix {
  heads: number;
  n: number;
  /** (heads, n, n) row-major, f16-decoded */
  data: Float32Array;
}

/** weight at (query i, key j) for one head, or the mean over heads. */
export function headAt(
  data: Float32Array,
  heads: number,
  n: number,
  head: number,
  i: number,
  j: number,
): number {
  if (head >= 0) return data[head * n * n + i * n + j];
  let sum = 0;
  for (let h = 0; h < heads; h++) sum += data[h * n * n + i * n + j];
  return sum / heads;
}

/** how much attention each key position receives, summed over the queries of
 * one row. row < 0 means the whole matrix, averaged over rows. */
export function keyMass(matrix: AttentionMatrix, head: number, row: number): Float32Array {
  const n = matrix.n;
  const out = new Float32Array(n);
  const lo = row < 0 ? 0 : row;
  const hi = row < 0 ? n - 1 : row;
  for (let i = lo; i <= hi; i++)
    for (let j = 0; j <= i; j++) out[j] += headAt(matrix.data, matrix.heads, n, head, i, j);
  if (row < 0) for (let j = 0; j < n; j++) out[j] /= n;
  return out;
}

export interface ResidDelta {
  layers: number;
  maxAbs: number;
  at: number;
  meanAbs: number;
}

export function residDelta(a: Float32Array, b: Float32Array): ResidDelta | null {
  const layers = Math.min(a.length, b.length);
  if (layers === 0) return null;
  let maxAbs = 0;
  let at = 0;
  let sum = 0;
  for (let l = 0; l < layers; l++) {
    const d = Math.abs(a[l] - b[l]);
    if (d > maxAbs) {
      maxAbs = d;
      at = l;
    }
    sum += d;
  }
  return { layers, maxAbs, at, meanAbs: sum / layers };
}

// -- the verdict: what did the edit actually do --

/** one place the two prompts differ, translated from characters to tokens.
 * the tokens come from the *new* prompt (b) where the char exists, which is
 * what the attention columns are indexed by. */
export interface EditEntry {
  kind: "insert" | "delete" | "replace";
  /** token index on the b prompt for insert/replace, on the a prompt for delete */
  tokenIndex: number;
  /** index in the b prompt, -1 when the chars have no counterpart there */
  bi: number;
  /** index in the a prompt, -1 when the chars have no counterpart there */
  ai: number;
  /** tokens the new chars are folded into */
  newTokenIds: number[];
  /** tokens the old chars came out of */
  oldTokenIds: number[];
}

export interface EditSummary {
  /** the raw character edit, as a short line */
  change: string;
  entries: EditEntry[];
  insertedTokens: number[];
  removedTokens: number[];
  /** a's token index -> b's token index, for tokens the edit left alone */
  align: number[];
  insertedChars: number;
  deletedChars: number;
}

/** character-level edit map: per char index, did it change, and what is it
 * aligned to. shared by the token edit summary and the attention shift report
 * so both agree on what "the same position" means. */
interface EditMap {
  changedA: boolean[];
  changedB: boolean[];
  alignA: number[];
  alignB: number[];
}

function editMap(va: string, vb: string): EditMap {
  const m = diffMarks(va, vb);
  return { changedA: m.changedA, changedB: m.changedB, alignA: m.alignA, alignB: m.alignB };
}

/** runs of unchanged chars only; everything else is an edit. */
interface SameRun {
  ai: number;
  bi: number;
  n: number;
}

function sameRuns(map: EditMap, va: string, vb: string): SameRun[] {
  const runs: SameRun[] = [];
  let i = 0;
  while (i < va.length) {
    if (map.changedA[i]) {
      i++;
      continue;
    }
    const bi = map.alignA[i];
    if (bi < 0 || bi >= vb.length) {
      i++;
      continue;
    }
    let n = 1;
    while (
      i + n < va.length &&
      !map.changedA[i + n] &&
      map.alignA[i + n] === bi + n &&
      !map.changedB[bi + n]
    ) {
      n++;
    }
    runs.push({ ai: i, bi, n });
    i += n;
  }
  return runs;
}

/** char index -> token index, per prompt. len = chars + 1 so a char index and
 * the "one past the end" index both resolve. */
function charToToken(prompt: string, tokenIds: number[], texts: Record<number, string>): number[] {
  const out = new Array<number>(prompt.length + 1).fill(-1);
  if (tokenIds.length === 0) return out;
  const pieces = tokenIds.map((id) => (texts[id] !== undefined ? texts[id] : ""));
  if (pieces.every((p) => p.length === 0)) {
    // texts have not decoded yet: assume the tokens tile the string evenly,
    // which is the best guess available and is replaced on the next render
    const per = Math.max(1, Math.ceil(prompt.length / tokenIds.length));
    for (let c = 0; c <= prompt.length; c++) {
      out[c] = Math.min(tokenIds.length - 1, Math.floor(c / per));
    }
    return out;
  }
  let c = 0;
  for (let t = 0; t < pieces.length && c <= prompt.length; t++) {
    const end = Math.min(prompt.length + 1, c + Math.max(1, pieces[t].length));
    for (; c < end; c++) out[c] = t;
  }
  for (; c <= prompt.length; c++) out[c] = tokenIds.length - 1;
  return out;
}

function tokenText(ids: number[], texts: Record<number, string>): string {
  const s = ids
    .map((id) => (texts[id] !== undefined ? visible(texts[id]) : `#${id}`))
    .join("");
  return s.length > 0 ? s : "∅";
}

/** the character diff expressed in tokens, with the token alignment the
 * attention columns need. */
export function editSummary(
  a: string,
  b: string,
  promptA: number[],
  promptB: number[],
  texts: Record<number, string>,
): EditSummary | null {
  const va = visibleForDiff(a);
  const vb = visibleForDiff(b);
  const map = editMap(va, vb);
  if (va === vb) return null;

  const ta = charToToken(va, promptA, texts);
  const tb = charToToken(vb, promptB, texts);
  const entries: EditEntry[] = [];
  let insertedChars = 0;
  let deletedChars = 0;

  // walk the changed chars in order, grouping the ones that land in the same
  // pair of tokens: one insertion inside a token is one entry, not one per char
  let i = 0;
  while (i < va.length || i < vb.length) {
    const aChanged = i < va.length && map.changedA[i];
    const bChanged = i < vb.length && map.changedB[i];
    if (!aChanged && !bChanged) {
      i++;
      continue;
    }
    const aTokens = new Set<number>();
    const bTokens = new Set<number>();
    let del = 0;
    let ins = 0;
    let k = i;
    while (k < Math.max(va.length, vb.length)) {
      const ac = k < va.length && map.changedA[k];
      const bc = k < vb.length && map.changedB[k];
      if (!ac && !bc) break;
      if (ac) {
        aTokens.add(ta[k]);
        del++;
      }
      if (bc) {
        bTokens.add(tb[k]);
        ins++;
      }
      k++;
    }
    i = k;
    insertedChars += ins;
    deletedChars += del;

    const isInsert = del === 0;
    const isDelete = ins === 0;
    const tokenIndex = [...(isDelete ? aTokens : bTokens)][0];
    if (tokenIndex === undefined) continue;
    entries.push({
      kind: isInsert ? "insert" : isDelete ? "delete" : "replace",
      tokenIndex,
      ai: [...aTokens][0] ?? -1,
      bi: [...bTokens][0] ?? -1,
      newTokenIds: [...bTokens].map((t) => promptB[t]).filter((v) => v !== undefined),
      oldTokenIds: [...aTokens].map((t) => promptA[t]).filter((v) => v !== undefined),
    });
  }
  if (entries.length === 0) return null;

  // token alignment across the unchanged chars. one-to-one on purpose: if a
  // token's chars map to more than one token on the other side it was split by
  // the edit, and "the same position" no longer exists for it
  const align = new Array<number>(promptA.length).fill(-1);
  const pairs = new Map<number, Set<number>>();
  for (const run of sameRuns(map, va, vb)) {
    for (let c = 0; c < run.n; c++) {
      const at = ta[run.ai + c];
      const bt = tb[run.bi + c];
      if (at < 0 || bt < 0) continue;
      const set = pairs.get(at) ?? new Set<number>();
      set.add(bt);
      pairs.set(at, set);
    }
  }
  for (const [at, bts] of pairs) {
    if (at < align.length && bts.size === 1) align[at] = [...bts][0];
  }

  return {
    change: describeEdit(insertedChars, deletedChars, entries, texts),
    entries,
    insertedTokens: [...new Set(entries.filter((e) => e.kind !== "delete").map((e) => e.tokenIndex))],
    removedTokens: [...new Set(entries.filter((e) => e.kind !== "insert").map((e) => e.tokenIndex))],
    align,
    insertedChars,
    deletedChars,
  };
}

function describeEdit(
  insertedChars: number,
  deletedChars: number,
  entries: EditEntry[],
  texts: Record<number, string>,
): string {
  const parts: string[] = [];
  for (const e of entries.slice(0, 3)) {
    const to = e.newTokenIds.length > 0 ? tokenText(e.newTokenIds, texts) : "∅";
    const from = e.oldTokenIds.length > 0 ? tokenText(e.oldTokenIds, texts) : "∅";
    if (e.kind === "insert") parts.push(`+“${to}” at token ${e.tokenIndex}`);
    else if (e.kind === "delete") parts.push(`−“${from}” at token ${e.tokenIndex}`);
    else parts.push(`“${from}”→“${to}” at token ${e.tokenIndex}`);
  }
  const count = `${insertedChars > 0 ? `+${insertedChars}` : ""}${deletedChars > 0 ? `−${deletedChars}` : ""} chars`;
  return `${count} · ${parts.join(", ")}${entries.length > 3 ? ` +${entries.length - 3} more` : ""}`;
}

/** attention mass at the first key position, averaged over queries. the
 * first-token sink: a change there is a change in what the model anchors on. */
export function firstKeyMass(matrix: AttentionMatrix, head: number): number {
  return keyMass(matrix, head, -1)[0] ?? 0;
}

/** perplexity of the key distribution: the effective number of keys attended.
 * lower = spikier. */
export function keyConcentration(mass: ArrayLike<number>): number {
  let total = 0;
  for (let i = 0; i < mass.length; i++) total += mass[i];
  if (total <= 0) return 0;
  return Math.exp(entropyNats(Array.from(mass, (v) => v / total)));
}

/** mass each key position receives, for one query row, normalised to sum to 1. */
export function rowMassOn(
  matrix: AttentionMatrix,
  head: number,
  i: number,
  out: Float32Array,
): void {
  const n = matrix.n;
  let total = 0;
  for (let j = 0; j < n; j++) {
    const w = j <= i ? headAt(matrix.data, matrix.heads, n, head, i, j) : 0;
    out[j] = w;
    total += w;
  }
  if (total > 0) for (let j = 0; j < n; j++) out[j] /= total;
}

export interface Shift {
  /** position in the a prompt */
  ai: number;
  /** its aligned position in b, or -1 when the char has no counterpart */
  bi: number;
  /** a's mass minus b's at the aligned position: positive = a attended more */
  delta: number;
  /** the position itself sits past the last token the edit touched, so its
   * change happened outside the edited span */
  beyondEdit: boolean;
}

/** where attention mass moved, biggest moves first. positions are matched
 * through the character alignment, so a shift caused by re-tokenization reads
 * as "this token gained / lost" instead of as an index shift. `boundary` is the
 * last position the edit touched: a move that lands past it left the edit's own
 * neighbourhood, which is the difference between re-tokenizing and re-deciding. */
export function attentionShifts(
  massA: ArrayLike<number>,
  massB: ArrayLike<number>,
  alignment: ArrayLike<number>,
  boundary = Number.POSITIVE_INFINITY,
): Shift[] {
  const out: Shift[] = [];
  for (let k = 0; k < massA.length; k++) {
    const bi = k < alignment.length ? alignment[k] : -1;
    const mb = bi >= 0 && bi < massB.length ? massB[bi] : 0;
    out.push({
      ai: k,
      bi,
      delta: massA[k] - mb,
      // this column sits past the edit on either side, so whatever happened to
      // it happened outside the edited span
      beyondEdit: Math.max(k, bi) > boundary,
    });
  }
  out.sort((x, y) => Math.abs(y.delta) - Math.abs(x.delta));
  return out;
}

/** attention mass landing on a set of key positions. */
export function massOn(idxs: readonly number[], mass: ArrayLike<number>): number {
  let sum = 0;
  for (const i of idxs) if (i >= 0 && i < mass.length) sum += mass[i];
  return sum;
}

/** the strongest weight anywhere in the square: how sure the head is about
 * anything at all, used as a "did the shape change" headline. */
export function attentionPeak(matrix: AttentionMatrix, head: number): number {
  let peak = 0;
  for (let i = 0; i < matrix.n; i++)
    for (let j = 0; j <= i; j++) {
      const w = headAt(matrix.data, matrix.heads, matrix.n, head, i, j);
      if (w > peak) peak = w;
    }
  return peak;
}

/** mean weight on the diagonal: how much each position looks at itself. */
export function selfAttention(matrix: AttentionMatrix, head: number): number {
  if (matrix.n === 0) return 0;
  let sum = 0;
  for (let i = 0; i < matrix.n; i++) sum += headAt(matrix.data, matrix.heads, matrix.n, head, i, i);
  return sum / matrix.n;
}

export interface AttentionMatrixPair {
  a: AttentionMatrix;
  b: AttentionMatrix;
  /** a's key position -> b's, or -1. from EditSummary. */
  align: number[];
  /** b key positions the edit created */
  created: readonly number[];
}

/** the key positions the two matrices can honestly be compared on: a key whose
 * identity survived the edit on both sides. a token the edit split has no
 * partner, so its mass is re-tokenization rather than a change of target —
 * which is the distinction the whole measurement rests on. */
interface KeyComparison {
  pairs: [number, number][];
  /** every a column the edit touched (split or dropped) */
  movedA: Set<number>;
  /** every b column the edit touched (created or left without a partner) */
  movedB: Set<number>;
}

function keyComparison(pair: AttentionMatrixPair, na: number, nb: number): KeyComparison {
  const created = new Set<number>(pair.created.filter((p) => p < nb));
  const pairs: [number, number][] = [];
  const movedA = new Set<number>();
  const movedB = new Set<number>(created);
  for (let k = 0; k < na; k++) {
    const bi = k < pair.align.length ? pair.align[k] : -1;
    if (bi < 0 || bi >= nb) {
      movedA.add(k);
      continue;
    }
    if (created.has(bi)) {
      // the edit folded this key's chars into a new token: the mass is local
      movedA.add(k);
      continue;
    }
    pairs.push([k, bi]);
  }
  for (let k = 0; k < nb; k++) {
    if (created.has(k)) continue;
    // a b column with no a partner was created by the split
    if (!pairs.some(([, b]) => b === k)) movedB.add(k);
  }
  return { pairs, movedA, movedB };
}

/** one row's attention over keys, normalised to sum to 1. */
function keyRow(matrix: AttentionMatrix, head: number, i: number, out: Float32Array): void {
  const n = matrix.n;
  let total = 0;
  for (let j = 0; j < n; j++) {
    const w = j <= i ? headAt(matrix.data, matrix.heads, n, head, i, j) : 0;
    out[j] = w;
    total += w;
  }
  if (total > 0) for (let j = 0; j < n; j++) out[j] /= total;
}

export interface AttentionChannels {
  /** how much attention mass changed target, as a share of the layer: the
   * size of the perturbation */
  redistributedMass: number;
  /** how many rows were re-drawn (TV >= 0.02), and out of how many comparable
   * rows */
  rowsChanged: number;
  rowsTotal: number;
  /** the row that changed most, and which one it was */
  rowMaxTv: number;
  rowMaxAt: number;
  /** of the mass that moved, how much stayed on the edit's own columns
   * (re-tokenization) and how much landed on untouched positions (mechanism) */
  localShare: number;
  nonLocalShare: number;
  /** rows whose biggest move was onto an untouched position */
  nonLocalRows: number;
  /** the two matrices were the same size, so the delta heatmap also applies */
  sameSize: boolean;
}

const ROW_CHANGED_TV = 0.02;
/** below this share of the layer, "where did the displaced mass go" is f16
 * quantization noise: the server ships attention as f16, and a 1e-4 cell
 * difference has no direction it can be trusted to point in. */
const MASS_NOISE_FLOOR = 0.005;

/** how far the edit travelled through the layer's attention. this is the
 * measurement the compare view exists for: a one-character edit can stay inside
 * the token it landed on, or re-draw attention across the context. rows and
 * columns are matched through the character alignment, so a re-tokenized prompt
 * is compared by identity rather than by index. */
export function compareAttentionChannels(
  pair: AttentionMatrixPair,
  head: number,
  edit: EditSummary,
): AttentionChannels | null {
  void edit;
  const { a, b } = pair;
  if (a.heads !== b.heads || a.n === 0 || b.n === 0) return null;
  const na = a.n;
  const nb = b.n;
  const rows = Math.min(na, nb);
  const { pairs, movedA, movedB } = keyComparison(pair, na, nb);
  if (pairs.length === 0) return null;

  const rowA = new Float32Array(na);
  const rowB = new Float32Array(nb);
  let tvTotal = 0;
  let nonLocalMass = 0;
  let localMass = 0;
  let rowsChanged = 0;
  let nonLocalRows = 0;
  let compared = 0;
  let rowMaxAt = -1;
  let rowMaxTv = 0;

  for (let i = 0; i < rows; i++) {
    keyRow(a, head, i, rowA);
    keyRow(b, head, i, rowB);
    // renormalise over the shared keys so the tail a row cannot have in common
    // does not read as a redistribution, and skip rows where almost nothing is
    // shared: those rows are entirely made of re-tokenization, not of a change
    // in what the model looks at
    let massA = 0;
    let massB = 0;
    for (const [ka, kb] of pairs) {
      massA += rowA[ka];
      massB += rowB[kb];
    }
    const norm = massA + massB;
    if (norm < 0.5) continue;
    compared++;

    let diff = 0;
    let nonLocal = 0;
    let local = 0;
    for (const [ka, kb] of pairs) {
      const pa = (2 * rowA[ka]) / norm;
      const pb = (2 * rowB[kb]) / norm;
      diff += Math.abs(pa - pb);
      const moved = Math.abs(pa - pb) / 2;
      if (movedB.has(kb)) local += moved;
      else nonLocal += moved;
    }
    // mass that left or entered the edit's own slots without a comparable
    // partner is the same bookkeeping at the column level
    let outOfEdit = 0;
    for (const k of movedA) if (k < na) outOfEdit += rowA[k];
    let intoEdit = 0;
    for (const k of movedB) if (k < nb) intoEdit += rowB[k];
    local += (Math.abs(outOfEdit - intoEdit) / Math.max(outOfEdit + intoEdit, 1e-9)) * 0.5;

    // over the renormalised pair both rows sum to 2, so the total-variation
    // distance between them is half the summed absolute difference
    const tv = diff / 2;
    tvTotal += tv;
    localMass += local;
    nonLocalMass += nonLocal;
    if (tv >= ROW_CHANGED_TV) {
      rowsChanged++;
      if (nonLocal > local) nonLocalRows++;
    }
    if (tv > rowMaxTv) {
      rowMaxTv = tv;
      rowMaxAt = i;
    }
  }

  const placed = localMass + nonLocalMass;
  if (compared === 0) return null;
  const redistributedMass = tvTotal / compared;
  // below this, "where did the mass go" is measuring f16 quantization noise
  // rather than a decision: report the split as nothing rather than as 70% of
  // 0.1%
  const meaningful = redistributedMass >= MASS_NOISE_FLOOR;
  return {
    // each row carries mass 1, so the mean TV is the share of the layer that
    // changed target
    redistributedMass,
    rowsChanged,
    rowsTotal: compared,
    rowMaxTv,
    rowMaxAt,
    localShare: meaningful && placed > 0 ? localMass / placed : 0,
    nonLocalShare: meaningful && placed > 0 ? nonLocalMass / placed : 0,
    nonLocalRows: meaningful ? nonLocalRows : 0,
    sameSize: na === nb,
  };
}
export interface Scope {
  kind: "global" | "local" | "mixed" | "none";
  /** the sentence the verdict can print */
  text: string;
}

/** local vs global: the judgement the attention-transfer analysis is after. a
 * perturbation is global when it re-draws rows away from the edit, not when it
 * merely re-splits the token it landed on. rows that only moved inside the
 * edited span do not count as reach. */
export function classifyScope(ch: AttentionChannels | null, edit: EditSummary): Scope {
  void edit;
  if (!ch || ch.rowsTotal === 0) return { kind: "none", text: "no attention pair to compare" };
  const pct = (x: number): string => `${(x * 100).toFixed(1)}%`;
  const broad = ch.nonLocalRows >= Math.max(2, Math.ceil(ch.rowsTotal * 0.1));
  if (ch.rowsChanged === 0 || (ch.nonLocalShare === 0 && ch.redistributedMass < 0.02)) {
    return {
      kind: "none",
      text: `attention holds: ${ch.rowsChanged}/${ch.rowsTotal} rows move past ${(ROW_CHANGED_TV * 100).toFixed(0)}%, and only ${pct(ch.redistributedMass)} of the layer's mass changes target`,
    };
  }
  if (broad && ch.nonLocalShare >= 0.4) {
    return {
      kind: "global",
      text: `global: ${ch.rowsChanged}/${ch.rowsTotal} rows are re-drawn and ${pct(ch.nonLocalShare)} of the displaced mass lands on positions the edit never touched`,
    };
  }
  if (ch.nonLocalRows <= 1 && ch.nonLocalShare < 0.25) {
    return {
      kind: "local",
      text: `local: ${pct(ch.redistributedMass)} of attention mass moves and ${pct(ch.localShare)} of it stays inside the edited span`,
    };
  }
  return {
    kind: "mixed",
    text: `mixed: ${ch.rowsChanged}/${ch.rowsTotal} rows are re-drawn, and ${pct(ch.nonLocalShare)} of the displaced mass reaches untouched positions`,
  };
}

// -- the conclusion: what the edit changed --

export type VerdictLevel = "high" | "mid" | "low";

export interface Observation {
  level: VerdictLevel;
  /** "edit" is the diff itself, reported for context; "effect" is something the
   * model did differently because of it. severity only counts effects. */
  kind: "edit" | "effect";
  /** one line of evidence */
  text: string;
}

export interface VerdictSeverity {
  level: "minimal" | "moderate" | "major";
  /** the short label for the badge */
  label: string;
}

export interface VerdictInput {
  /** a's explicit token ids, for naming positions the capture said nothing about */
  promptA: number[];
  promptB: number[];
  texts: Record<number, string>;
  /** "a" and "b" unless the columns are drawn swapped */
  tagA: string;
  tagB: string;
  /** how to name the stack in a sentence, e.g. "the lens on both sides" */
  lensLabel: string;
  /** what the numbers cover: "L0 / 30 · mean head". the verdict is only true of
   * this slice, and saying so is what keeps a single-layer reading from being
   * mistaken for the whole model. */
  measured: string;
  edit: EditSummary;
  /** how the layer's attention redistributed; null without an attention pair */
  channels: AttentionChannels | null;
  nextTop: {
    idA: number;
    idB: number;
    pA: number;
    pB: number;
    js: number;
    tv: number;
    sameTop: boolean;
  } | null;
  attention: {
    head: number;
    massA: Float32Array;
    massB: Float32Array;
    stats: AttentionStats | null;
    /** largest weight in each square */
    peakA: number;
    peakB: number;
    selfA: number;
    selfB: number;
  } | null;
  resid: ResidDelta | null;
  /** the residual norm scale, for a relative "did the stream move" call */
  residScale: number;
  lens: {
    diverge: number | null;
    agree: number;
    layers: number;
    /** first layer where the two sides' top-1 differ (null if none did) */
    firstKeyShift: number | null;
    /** last such layer, so "reconverged" can be told from "never did" */
    lastKeyShift: number | null;
  } | null;
}

export interface Verdict {
  /** the edit itself, in tokens */
  change: string;
  /** whether the perturbation stayed local or re-drew the layer */
  scope: Scope;
  /** what the numbers cover, e.g. "L0 / 30 · mean head" */
  measured: string;
  /** the closing line: the answer, stated before the evidence */
  conclusion: string;
  /** the same answer as one clause, for the panel headline */
  answer: string;
  observations: Observation[];
  /** bullet text, also what the copy button puts on the clipboard */
  text: string;
  severity: VerdictSeverity;
}

/** turn every measured difference into the plain-language answer to "what
 * happened when this prompt became that one". thresholds are deliberately
 * loose: the point is to stop the reader from having to eyeball four panels to
 * find out that nothing moved. */
export function buildVerdict(input: VerdictInput): Verdict {
  const { tagA, tagB, edit } = input;
  const obs: Observation[] = [];
  const name = (tag: string, i: number): string =>
    `${tag}@${i}${tokenName(input, i, tag) ? ` “${tokenName(input, i, tag)}”` : ""}`;

  // 1. the edit itself, first: everything below is read relative to it. tagged
  // "edit" so it cannot inflate the severity — it is the question, not an
  // answer.
  for (const e of edit.entries.slice(0, 3)) {
    const label = (ids: number[]): string => (ids.length > 0 ? tokenText(ids, input.texts) : "∅");
    if (e.kind === "insert") {
      obs.push({
        level: "low",
        kind: "edit",
        text: `inserted “${label(e.newTokenIds)}” at token ${e.tokenIndex}`,
      });
    } else if (e.kind === "delete") {
      obs.push({
        level: "low",
        kind: "edit",
        text: `deleted “${label(e.oldTokenIds)}” at token ${e.tokenIndex}`,
      });
    } else {
      obs.push({
        level: "low",
        kind: "edit",
        text: `token ${e.ai}→${e.bi} replaced: “${label(e.oldTokenIds)}”→“${label(e.newTokenIds)}”`,
      });
    }
  }
  const gained = new Set(edit.insertedTokens);
  const lost = new Set(edit.removedTokens);
  const scope = classifyScope(input.channels, edit);
  if (input.channels && input.attention) {
    // 2. the headline of the mechanism: how far a one-character perturbation got
    const ch = input.channels;
    obs.push({
      level:
        scope.kind === "global"
          ? "high"
          : ch.rowsChanged >= 2 || ch.redistributedMass >= 0.02
            ? "mid"
            : "low",
      kind: "effect",
      text: `perturbation: ${(ch.redistributedMass * 100).toFixed(1)}% of the layer's attention mass changed target, ${ch.rowsChanged}/${ch.rowsTotal} query rows re-drawn (worst row ${(ch.rowMaxTv * 100).toFixed(0)}%${ch.rowMaxAt >= 0 ? ` at position ${ch.rowMaxAt}` : ""})`,
    });
    if (ch.nonLocalRows > 0 || ch.nonLocalShare > 0.05) {
      obs.push({
        level: ch.nonLocalRows >= Math.max(2, Math.ceil(ch.rowsTotal * 0.1)) ? "high" : "mid",
        kind: "effect",
        text: `reach: ${ch.nonLocalRows}/${ch.rowsTotal} rows mostly attend somewhere the edit never touched, and ${(ch.nonLocalShare * 100).toFixed(1)}% of the displaced mass lands there (${(ch.localShare * 100).toFixed(1)}% stays inside the edited span)`,
      });
    }
  }

  if (input.attention) {
    const at = input.attention;
    // measure the relocation on positions that survived the edit with a single
    // counterpart. a token the edit split has no aligned position by
    // definition, and counting its whole mass as "moved" would drown out the
    // real signal — the split is reported on its own below.
    const comparable = attentionShifts(at.massA, at.massB, edit.align, editBoundary(edit)).filter(
      (s) => s.bi >= 0 && !gained.has(s.ai) && !lost.has(s.ai),
    );
    const top = comparable[0];
    const totalMoved = comparable.reduce((s, x) => s + Math.abs(x.delta), 0);

    // 3. where mass moved: the actual answer to "how did attention shift"
    if (top && Math.abs(top.delta) >= 0.004) {
      const editAt = edit.entries[0].tokenIndex;
      const at = top.ai > editAt ? `${top.ai - editAt} past the edit` : "at the edit";
      let inSpan = 0;
      let outSpan = 0;
      for (const s of comparable) {
        const moved = Math.abs(s.delta);
        if (s.beyondEdit) outSpan += moved;
        else inSpan += moved;
      }
      obs.push({
        level: Math.abs(top.delta) >= 0.03 ? "high" : "mid",
        kind: "effect",
        text: `largest move: ${name(tagA, top.ai)} −${Math.abs(top.delta).toFixed(4)} → ${name(tagB, top.bi)} +same (${at}). Across comparable positions ${totalMoved.toFixed(4)} is re-drawn: ${inSpan.toFixed(4)} on columns inside the edited span, ${outSpan.toFixed(4)} on columns past it`,
      });
      if (comparable[1] && Math.abs(comparable[1].delta) >= 0.004) {
        const s = comparable[1];
        obs.push({
          level: "low",
          kind: "effect",
          text: `next largest move: ${name(tagA, s.ai)} −${Math.abs(s.delta).toFixed(4)} → ${name(tagB, s.bi)} +same`,
        });
      }
    } else {
      obs.push({
        level: "low",
        kind: "effect",
        text: `no untouched position loses more than ${(top ? Math.abs(top.delta) : 0).toFixed(4)}: the edit's own slot absorbed it`,
      });
    }

    // 4. the peak: what the model is most sure about, and whether it moved
    const peakMoved =
      at.stats && at.stats.maxAbs >= 0.02
        ? `largest cell change at the peak ${at.stats.maxAbs.toFixed(4)} @ (${at.stats.at?.i ?? "–"},${at.stats.at?.j ?? "–"})`
        : `the peak did not move (largest cell change ${(at.stats?.maxAbs ?? 0).toFixed(4)})`;
    obs.push({
      level: at.stats && at.stats.maxAbs >= 0.1 ? "high" : "low",
      kind: "effect",
      text: `strongest weight ${at.peakA.toFixed(3)} → ${at.peakB.toFixed(3)}; ${peakMoved}`,
    });

    // 5. what happens to the mass that used to sit on one token
    if (gained.size > 0) {
      const newMass = massOn([...gained], at.massB);
      const total = at.massB.reduce((s, v) => s + v, 0);
      const share = total > 0 ? newMass / total : 0;
      obs.push({
        // a new token that walks off with a real share of the row's attention is
        // a shift in where the model looks; a sliver is just re-tokenization
        level: share >= 0.15 ? "mid" : "low",
        kind: "effect",
        text: `re-tokenization: the new token ${[...gained].map((i) => name(tagB, i)).join(", ")} holds ${(share * 100).toFixed(1)}% of that row's attention`,
      });
    }
    if (lost.size > 0) {
      obs.push({
        level: "mid",
        kind: "effect",
        text: `vanished token ${[...lost].map((i) => name(tagA, i)).join(", ")} held ${massOn([...lost], at.massA).toFixed(4)}`,
      });
    }

    // 6. first key: the anchor
    const firstA = at.massA[0] ?? 0;
    const firstB = at.massB[0] ?? 0;
    if (Math.abs(firstB - firstA) >= 0.01) {
      obs.push({
        level: "mid",
        kind: "effect",
        text: `first-token attention ${firstA.toFixed(4)} → ${firstB.toFixed(4)}: the anchor changed`,
      });
    }

    // 6. spread: is the model looking at fewer or more places
    const spreadA = keyConcentration(at.massA);
    const spreadB = keyConcentration(at.massB);
    if (spreadA > 0 && spreadB > 0 && Math.abs(spreadB - spreadA) / spreadA >= 0.03) {
      obs.push({
        level: "low",
        kind: "effect",
        text: `attention spread ${spreadA.toFixed(2)} → ${spreadB.toFixed(2)} effective positions (${spreadB < spreadA ? "more focused" : "more diffuse"})`,
      });
    }

    // 7. self attention
    if (Math.abs(at.selfB - at.selfA) >= 0.01) {
      obs.push({
        level: "low",
        kind: "effect",
        text: `mean self-attention ${at.selfA.toFixed(4)} → ${at.selfB.toFixed(4)}`,
      });
    }
  } else {
    obs.push({
      level: "low",
      kind: "effect",
      text: "no attention matrices (different context lengths, or attn is off) — numbers only",
    });
  }

  // 8. next token
  if (input.nextTop) {
    const n = input.nextTop;
    if (!n.sameTop) {
      obs.push({
        level: "high",
        kind: "effect",
        text: `the next token flips: ${tagA} picks ${tokenText([n.idA], input.texts)} (${fmtPct(n.pA)}), ${tagB} picks ${tokenText([n.idB], input.texts)} (${fmtPct(n.pB)})`,
      });
    } else {
      obs.push({
        level: "low",
        kind: "effect",
        text: `the next token holds: both pick ${tokenText([n.idA], input.texts)} (${fmtPct(n.pA)} → ${fmtPct(n.pB)})`,
      });
    }
    obs.push({
      level: n.js >= 0.05 ? "high" : n.js >= 0.01 ? "mid" : "low",
      kind: "effect",
      text: `distribution moved JS ${n.js.toFixed(4)} nats (TV ${n.tv.toFixed(4)})`,
    });
  }

  // 9. the stack: how deep the perturbation reached, and whether it survived
  if (input.lens) {
    const l = input.lens;
    if (l.diverge === null) {
      obs.push({
        level: "low",
        kind: "effect",
        text: `logit lens: all ${l.layers} layers agree on the argmax — the perturbation never reaches the decision`,
      });
    } else {
      const from = l.diverge === 0 ? "emb" : `L${l.diverge - 1}`;
      const first = l.firstKeyShift === null ? l.diverge : l.firstKeyShift;
      const open = first === 0 ? "emb" : `L${first - 1}`;
      const reconverges = l.lastKeyShift !== null && l.lastKeyShift < l.layers - 1;
      obs.push({
        level: l.diverge <= 1 ? "high" : "mid",
        kind: "effect",
        text: `logit lens: the layer-wise top-1 first splits at ${open}${l.diverge > first ? ` (argmax differs from ${from})` : ""} and ${reconverges ? `re-converges by L${l.lastKeyShift}` : "never reconverges"}; ${l.agree}/${l.layers} layers agree`,
      });
    }
  }
  if (input.resid) {
    const scale = input.residScale > 0 ? input.residScale : 1;
    const rel = input.resid.maxAbs / scale;
    obs.push({
      level: rel >= 0.25 ? "high" : rel >= 0.05 ? "mid" : "low",
      kind: "effect",
      text: `residual stream ‖h‖ diverges most at ${input.resid.at === 0 ? "emb" : `L${input.resid.at - 1}`}: |Δ| ${input.resid.maxAbs.toFixed(2)} (${(rel * 100).toFixed(1)}% of that layer's scale, mean ${input.resid.meanAbs.toFixed(2)})`,
    });
  }

  const severity = judgeSeverity(obs);
  const { conclusion, answer } = synthesize(input, severity, scope);
  // the panel lists them in severity order, edit facts first
  const rank: Record<VerdictLevel, number> = { high: 0, mid: 1, low: 2 };
  const sorted = [...obs].sort(
    (x, y) => (x.kind === y.kind ? rank[x.level] - rank[y.level] : x.kind === "edit" ? -1 : 1),
  );
  // answer first, then what it means, then the evidence: a reader should not
  // have to reach the bottom of the panel to find out what it concluded
  const text = [
    answer,
    conclusion,
    "",
    `measured: ${input.measured}`,
    `edit: ${edit.change}`,
    `scope: ${scope.text}`,
    "evidence:",
    ...sorted.map((o) => `- ${o.text}`),
  ].join("\n");
  return {
    change: edit.change,
    scope,
    measured: input.measured,
    conclusion,
    answer,
    observations: sorted,
    text,
    severity,
  };
}

function tokenName(input: VerdictInput, i: number, tag: string): string {
  const ids = tag === input.tagB ? input.promptB : input.promptA;
  const id = ids[i];
  if (id === undefined) return "";
  const t = input.texts[id];
  return t !== undefined ? visible(t) : `#${id}`;
}

function fmtPct(p: number): string {
  return p >= 0.0001 ? `${(p * 100).toFixed(1)}%` : "<0.01%";
}

/** only effects count toward severity: the edit line is always there, and
 * "there is a difference" is not itself a finding. */
function judgeSeverity(obs: Observation[]): VerdictSeverity {
  const effects = obs.filter((o) => o.kind === "effect");
  const high = effects.filter((o) => o.level === "high").length;
  const mid = effects.filter((o) => o.level === "mid").length;
  if (high >= 2 || (high === 1 && mid >= 1)) return { level: "major", label: "material change" };
  if (high === 1 || mid >= 2) return { level: "moderate", label: "measurable, limited" };
  return { level: "minimal", label: "basically unchanged" };
}

/** the answer, stated first and then justified. the view exists to ask one
 * question — can a single character (usually a whitespace) perturb the model
 * globally, or does it only re-tokenize what it landed on — so `answer` is that
 * question answered in one line, and `conclusion` says what it means for the
 * model. the numbers live in the evidence, not in the sentence. */
function synthesize(
  input: VerdictInput,
  severity: VerdictSeverity,
  scope: Scope,
): { conclusion: string; answer: string } {
  const ch = input.channels;

  // the verdict: the token re-split, or the layer re-drawn
  const reach =
    scope.kind === "global"
      ? "the whole layer is re-drawn"
      : scope.kind === "mixed"
        ? "the shift leaks past the edit's own token"
        : ch && ch.rowsChanged > 0
          ? "the shift stays inside the edit's own token"
          : "attention barely shifts at all";
  const answer = `Does one character perturb the layer globally? ${
    scope.kind === "global" ? "Yes" : scope.kind === "mixed" ? "Partly" : "No"
  } — ${reach} (${input.measured})`;

  // what it means for the model, when something moved
  const consequence =
    input.nextTop && input.nextTop.js >= 0.01
      ? input.nextTop.sameTop
        ? "the next-token distribution shifts"
        : "the next-token choice flips"
      : null;
  const point =
    input.lens && input.lens.diverge !== null
      ? `the lens starts splitting at ${input.lens.diverge === 0 ? "emb" : `L${input.lens.diverge - 1}`}`
      : null;
  const state = [point, consequence].filter(Boolean).join(" and ");

  if (severity.level === "minimal") {
    return { answer, conclusion: "the model reads this sentence the same way" };
  }
  const conclusion = state ? `${state} anyway` : "the model's next move is unchanged";
  return { answer, conclusion };
}

/** the last prompt position the edit touched, on either side. anything past it
 * is outside the edited span. */
function editBoundary(edit: EditSummary): number {
  let last = -1;
  for (const e of edit.entries) last = Math.max(last, e.ai, e.bi);
  return last;
}

