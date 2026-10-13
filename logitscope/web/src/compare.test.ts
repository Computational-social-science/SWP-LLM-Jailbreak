import { describe, expect, it } from "vitest";
import {
  attentionPeak,
  attentionShifts,
  attentionStats,
  buildVerdict,
  classifyScope,
  compareAttentionChannels,
  compareDist,
  compareDistributions,
  diffMarks,
  diffStrings,
  editSummary,
  keyConcentration,
  keyMass,
  massOn,
  probIn,
  residDelta,
  selfAttention,
  unionCandidates,
  type CompareSideCapture,
} from "./compare";
import { softmax } from "./math";

// -- prompt diff --

describe("diffStrings", () => {
  const sameCount = (ops: ReturnType<typeof diffStrings>): number =>
    ops.filter((o) => o.kind === "same").length;

  it("identical strings are all same-ops", () => {
    const ops = diffStrings("abc", "abc");
    expect(ops).toHaveLength(3);
    expect(sameCount(ops)).toBe(3);
  });

  it("a substitution is one del followed by one ins", () => {
    const ops = diffStrings("abc", "axc");
    expect(ops.map((o) => o.kind)).toEqual(["same", "del", "ins", "same"]);
    expect(sameCount(ops)).toBe(2);
  });

  it("an inserted whitespace shifts the tail but keeps it aligned", () => {
    const ops = diffStrings("a b", "a  b");
    const ins = ops.filter((o) => o.kind === "ins");
    expect(ins).toHaveLength(1);
    expect(sameCount(ops)).toBe(3);
  });

  it("a wholesale replacement falls back to del/ins past the cell budget", () => {
    const ops = diffStrings("abcd", "wxyz", 1);
    expect(ops.filter((o) => o.kind === "del")).toHaveLength(4);
    expect(ops.filter((o) => o.kind === "ins")).toHaveLength(4);
  });
});

describe("diffMarks", () => {
  it("flags the inserted space and nothing else", () => {
    const m = diffMarks("the cat", "the  cat");
    expect(m.changedA).toEqual(new Array(7).fill(false));
    expect(m.changedB.filter(Boolean)).toHaveLength(1);
    expect(m.changedB[3]).toBe(true); // the extra space
    expect(m.alignA[3]).toBe(4); // 'c' of a lines up with 'c' of b at 4
    expect(m.insRuns).toEqual([[3, 3]]);
    expect(m.delRuns).toEqual([]);
  });

  it("marks both sides when a character is replaced", () => {
    // the lcs keeps the second 'p', so the first one reads as deleted and the
    // 'x' as a fresh insert. a minimal edit script, just not a paired one: the
    // two edits are separated by a char the lcs matched.
    const m = diffMarks("apple", "apxle");
    expect(m.changedA).toEqual([false, true, false, false, false]);
    expect(m.changedB).toEqual([false, false, true, false, false]);
    expect(m.alignA[1]).toBe(-1);
    expect(m.alignB[2]).toBe(-1);
    expect(m.delRuns).toEqual([[1, 1]]);
    expect(m.insRuns).toEqual([[2, 2]]);
  });

  it("pairs an adjacent delete+insert as one replacement", () => {
    const m = diffMarks("ab", "ax");
    expect(m.changedA).toEqual([false, true]);
    expect(m.changedB).toEqual([false, true]);
    expect(m.alignA[1]).toBe(1);
    expect(m.alignB[1]).toBe(1);
    expect(m.delRuns).toEqual([[1, 1]]);
    expect(m.insRuns).toEqual([[1, 1]]);
  });

  it("a deleted character has no counterpart", () => {
    const m = diffMarks("ab", "a");
    expect(m.changedA[1]).toBe(true);
    expect(m.alignA[1]).toBe(-1);
    expect(m.changedB).toEqual([false]);
    expect(m.alignB.includes(1)).toBe(false); // nothing points at the delete
    expect(m.delRuns).toEqual([[1, 1]]);
  });

  it("runs of edits collapse into contiguous ranges", () => {
    const m = diffMarks("x123y", "x9y");
    expect(m.delRuns).toEqual([[1, 3]]);
    expect(m.insRuns).toEqual([[1, 1]]);
    expect(m.changedA.slice(1, 4)).toEqual([true, true, true]);
    expect(m.changedB[1]).toBe(true);
  });

  it("identical prompts mark nothing", () => {
    const m = diffMarks("same", "same");
    expect(m.changedA.some(Boolean)).toBe(false);
    expect(m.changedB.some(Boolean)).toBe(false);
    expect(m.delRuns).toEqual([]);
    expect(m.insRuns).toEqual([]);
  });

  it("only ever aligns unchanged chars to each other", () => {
    // the alignment table is what the ui uses to lay the two rows out against
    // each other, so an aligned pair that has a change on either side would be
    // a lie about what was compared
    const cases: [string, string][] = [
      ["the scope hums: ", "the  scope hums: "],
      ["apple", "apxle"],
      ["a b c", "abc "],
      ["", "x"],
      ["x", ""],
      ["\u3042 \u3044", "\u3042  \u3044"],
    ];
    for (const [a, b] of cases) {
      const m = diffMarks(a, b);
      for (let i = 0; i < a.length; i++) {
        const j = m.alignA[i];
        if (j < 0) continue;
        expect(m.alignB[j]).toBe(i);
        if (!m.changedA[i]) expect(m.changedB[j]).toBe(false);
      }
    }
  });
});

// -- next-token distributions --

function side(over: Partial<CompareSideCapture> = {}): CompareSideCapture {
  return {
    topkIds: new Uint32Array([1, 2, 3]),
    topkLogits: new Float32Array([2, 0, -2]),
    lens: null,
    residNorms: null,
    ...over,
  };
}

describe("compareDist", () => {
  it("softmaxes the top-k logits and reads the entropy back", () => {
    const d = compareDist(side())!;
    const want = softmax(new Float32Array([2, 0, -2]));
    expect(d.ids).toEqual([1, 2, 3]);
    for (let i = 0; i < 3; i++) expect(d.probs[i]).toBeCloseTo(want[i], 6);
    let h = 0;
    for (const p of want) h -= p * Math.log(p);
    expect(d.entropy).toBeCloseTo(h, 6);
  });

  it("takes the final lens layer as the model's own top pick", () => {
    const d = compareDist(
      side({
        lens: {
          layers: 2,
          k: 2,
          ids: new Uint32Array([7, 8, 9, 10]),
          probs: new Float32Array([0.1, 0.2, 0.3, 0.4]),
        },
      }),
    )!;
    expect(d.lensTop?.id).toBe(9);
    expect(d.lensTop?.p).toBeCloseTo(0.3, 6);
  });

  it("is null without candidates", () => {
    expect(compareDist(side({ topkIds: null, topkLogits: null }))).toBeNull();
    expect(compareDist(side({ topkIds: new Uint32Array(0) }))).toBeNull();
  });

  it("keeps a long tail alive instead of underflowing it to zero", () => {
    // the naive exp(logit)/sum flushes -200 and -400 to exactly 0 in f32
    const d = compareDist(
      side({ topkIds: new Uint32Array([1, 2, 3]), topkLogits: new Float32Array([0, -200, -400]) }),
    )!;
    expect(probIn(d, 2)).toBeGreaterThan(0);
    expect(probIn(d, 3)).toBeGreaterThan(0);
    expect(probIn(d, 2)).toBeLessThan(probIn(d, 1));
    // and p(id) agrees with the stored probability where both are representable
    expect(probIn(d, 1)).toBeCloseTo(d.probs[0], 12);
    expect(probIn(d, 99)).toBe(0);
    expect(probIn(null, 1)).toBe(0);
  });
});

describe("unionCandidates", () => {
  it("orders by the better rank on either side", () => {
    const a = compareDist(side({ topkIds: new Uint32Array([5, 6]) }))!;
    const b = compareDist(side({ topkIds: new Uint32Array([6, 7]) }))!;
    expect(unionCandidates(a, b, 3)).toEqual([5, 6, 7]);
    expect(unionCandidates(a, b, 2)).toEqual([5, 6]);
  });
});

describe("compareDistributions", () => {
  it("measures how far one distribution moved", () => {
    const a = compareDist(side({ topkIds: new Uint32Array([1]), topkLogits: new Float32Array([0]) }))!;
    const b = compareDist(side({ topkIds: new Uint32Array([1]), topkLogits: new Float32Array([0]) }))!;
    const m = compareDistributions(a, b);
    expect(m.tv).toBeCloseTo(0, 6);
    expect(m.js).toBeCloseTo(0, 6);
    expect(m.kl).toBeCloseTo(0, 6);
    expect(m.sameTop).toBe(true);
    expect(m.maxDelta?.id).toBe(1);
  });

  it("two one-hot distributions over different tokens are maximally apart", () => {
    // a lone candidate softmaxes to probability 1
    const a = compareDist(side({ topkIds: new Uint32Array([1]), topkLogits: new Float32Array([0]) }))!;
    const b = compareDist(side({ topkIds: new Uint32Array([2]), topkLogits: new Float32Array([0]) }))!;
    const m = compareDistributions(a, b);
    expect(m.tv).toBeCloseTo(1, 6);
    expect(m.kl).toBeNull(); // p>0 where q=0, so KL is undefined
    expect(m.js).toBeCloseTo(Math.log(2), 6); // half of KL(one-hot || mean) twice
    expect(m.sameTop).toBe(false);
    expect(m.maxDelta?.id).toBe(1);
  });
});

// -- attention --

describe("attentionStats", () => {
  // one head, n = 2, stored row-major as a full 2x2 square: only (0,0), (1,0)
  // and (1,1) are ever filled, (0,1) stays 0 by construction
  const m = (a11: number, a21: number, a22: number) => ({
    heads: 1,
    n: 2,
    data: new Float32Array([a11, 0, a21, a22]),
  });

  it("finds the biggest per-cell move and the peaks", () => {
    const st = attentionStats(m(0.5, 0.1, 0.4), m(0.2, 0.1, 0.8), -1)!;
    expect(st.maxAbs).toBeCloseTo(0.4, 6); // (1,1): 0.4 vs 0.8
    expect(st.at).toEqual({ i: 1, j: 1 });
    expect(st.maxA).toBeCloseTo(0.5, 6);
    expect(st.maxB).toBeCloseTo(0.8, 6);
    expect(st.meanAbs).toBeCloseTo((0.3 + 0 + 0.4) / 3, 6);
  });

  it("refuses mismatched shapes", () => {
    const wide = { heads: 1, n: 3, data: new Float32Array(9) };
    expect(attentionStats(m(1, 0, 1), wide, -1)).toBeNull();
  });
});

describe("keyMass", () => {
  const mat = { heads: 1, n: 2, data: new Float32Array([0.25, 0, 0.5, 0.25]) };

  it("sums attention per key over one query row", () => {
    const row1 = keyMass(mat, -1, 1);
    expect(Array.from(row1)).toEqual([0.5, 0.25]);
  });

  it("averages over all rows when row < 0", () => {
    const all = keyMass(mat, -1, -1);
    expect(all[0]).toBeCloseTo((0.25 + 0.5) / 2, 6); // 0.375
    expect(all[1]).toBeCloseTo(0.125, 6);
  });
});

describe("residDelta", () => {
  it("reports where the stream diverges most", () => {
    const d = residDelta(new Float32Array([1, 2, 3]), new Float32Array([1, 2.5, 3]))!;
    expect(d.maxAbs).toBeCloseTo(0.5, 6);
    expect(d.at).toBe(1);
    expect(d.meanAbs).toBeCloseTo(0.5 / 3, 6);
  });

  it("stops at the shorter capture", () => {
    const d = residDelta(new Float32Array([1, 2]), new Float32Array([1, 2, 3]))!;
    expect(d.layers).toBe(2);
  });

  it("is null when a side has no norms", () => {
    expect(residDelta(new Float32Array(0), new Float32Array([1]))).toBeNull();
  });
});

// -- the verdict --

describe("editSummary", () => {
  // "ab c" tokenizes as ["ab", " ", "c"] and "a b c" as ["a", " ", "b", " ", "c"];
  // the texts map is what lets the summary line the chars up with the tokens
  const texts: Record<number, string> = { 1: "ab", 2: " ", 3: "c", 4: "a", 5: "b" };
  const pA = [1, 2, 3];
  const pB = [4, 2, 5, 2, 3];

  it("reports an inserted char in token terms", () => {
    const e = editSummary("ab c", "a b c", pA, pB, texts)!;
    expect(e.insertedChars).toBe(1);
    expect(e.deletedChars).toBe(0);
    // the new space is its own token in b (index 1), which is where the
    // attention columns will show the new mass landing
    expect(e.insertedTokens).toEqual([1]);
    expect(e.removedTokens).toEqual([]);
    expect(e.entries[0].kind).toBe("insert");
    expect(e.change).toContain("+1 char");
    expect(e.change).toContain("at token 1");
  });

  it("aligns the unchanged tokens across the change", () => {
    const e = editSummary("ab c", "a b c", pA, pB, texts)!;
    // a's " " (token 1) is b's " " (token 3), and a's "c" (2) is b's "c" (4)
    expect(e.align[1]).toBe(3);
    expect(e.align[2]).toBe(4);
    // a's token 0 was split by the insert, so it has no single counterpart
    expect(e.align[0]).toBe(-1);
  });

  it("returns null when the prompts are identical", () => {
    expect(editSummary("ab c", "ab c", pA, pA, texts)).toBeNull();
  });

  it("pairs a replacement in the same token", () => {
    const e = editSummary("ab c", "ab d", pA, [1, 2, 6], { ...texts, 6: "d" })!;
    expect(e.insertedChars).toBe(1);
    expect(e.deletedChars).toBe(1);
    expect(e.entries).toHaveLength(1);
    expect(e.entries[0].kind).toBe("replace");
    // the replaced token survives on both sides, so it stays aligned
    expect(e.align[0]).toBe(0);
  });
});

describe("attention shifts and shape", () => {
  it("matches positions through the alignment instead of by index", () => {
    // a: [0.1, 0.4, 0.5], b: [0.1, 0.5] where a's token 1 and 2 both map to b's 1
    const shifts = attentionShifts(
      new Float32Array([0.1, 0.4, 0.5]),
      new Float32Array([0.1, 0.2]),
      [0, 1, 1],
    );
    expect(shifts[0].ai).toBe(2);
    expect(shifts[0].bi).toBe(1);
    expect(shifts[0].delta).toBeCloseTo(0.3, 6);
    expect(shifts[1].ai).toBe(1);
    expect(shifts[1].delta).toBeCloseTo(0.2, 6);
    expect(shifts[2].ai).toBe(0);
    expect(shifts[2].delta).toBeCloseTo(0, 6);
  });

  it("treats a position with no counterpart as fully lost", () => {
    // b lost its middle key entirely, so a's middle pos has nowhere to go
    const shifts = attentionShifts(new Float32Array([0.3, 0.2, 0.5]), new Float32Array([0.3, 0.5]), [
      0,
      -1,
      1,
    ]);
    const lost = shifts.find((s) => s.ai === 1)!;
    expect(lost.bi).toBe(-1);
    expect(lost.delta).toBeCloseTo(0.2, 6);
  });

  it("concentration is the effective number of keys", () => {
    expect(keyConcentration(new Float32Array([0.5, 0.5]))).toBeCloseTo(2, 6);
    expect(keyConcentration(new Float32Array([1, 0]))).toBeCloseTo(1, 6);
    expect(keyConcentration(new Float32Array([0, 0]))).toBe(0);
  });

  it("peaks and self-attention read the square", () => {
    const mat = { heads: 1, n: 2, data: new Float32Array([0.25, 0, 0.5, 0.75]) };
    expect(attentionPeak(mat, -1)).toBeCloseTo(0.75, 6); // max over {0.25, 0.5, 0.75}
    expect(selfAttention(mat, -1)).toBeCloseTo(0.5, 6); // mean of the diagonal 0.25, 0.75
    expect(massOn([0, 5, -1], new Float32Array([0.25, 0.75]))).toBeCloseTo(0.25, 6);
  });
});

describe("attention channels and scope", () => {
  // a: [10, 11, 12]; b inserts 21 between 10 and 11 -> [10, 21, 11, 12]
  const texts: Record<number, string> = { 10: "x", 11: "y", 12: "z", 21: " " };
  const pA = [10, 11, 12];
  const pB = [10, 21, 11, 12];
  const edit = editSummary("xyz", "x yz", pA, pB, texts)!;

  /** square matrix from per-row key distributions. rows are (key -> weight). */
  const matrix = (rows: [number, number][][], n: number) => {
    const data = new Float32Array(n * n);
    rows.forEach((row, i) => {
      for (const [j, w] of row) data[i * n + j] = w;
    });
    return { heads: 1, n, data };
  };

  it("calls a perturbation that re-draws rows away from the edit global", () => {
    // a: every row sits on one key. b: the insert shifts two rows off the key
    // they were on and onto a new one, which is the whole question.
    const a = matrix([[[0, 1]], [[1, 1]], [[2, 1]]], 3);
    const b = matrix([[[0, 1]], [[2, 0.3], [3, 0.7]], [[1, 0.2], [3, 0.8]], [[3, 1]]], 4);
    const ch = compareAttentionChannels(
      { a, b, align: edit.align, created: edit.insertedTokens },
      -1,
      edit,
    )!;
    expect(ch.rowsTotal).toBe(3); // only the rows both prompts have
    expect(ch.rowsChanged).toBeGreaterThanOrEqual(2);
    expect(ch.nonLocalRows).toBeGreaterThanOrEqual(2);
    expect(ch.nonLocalShare).toBeGreaterThan(0.4);
    expect(ch.redistributedMass).toBeGreaterThan(0.05);
    expect(ch.sameSize).toBe(false);
    expect(classifyScope(ch, edit).kind).toBe("global");
  });

  it("calls a perturbation that only renumbers a slot a non-event", () => {
    // the two prompts have the same attention rows, so nothing moved between
    // identities: the insert renumbered a slot but did not redirect anything
    const a = matrix([[[0, 1]], [[1, 1]], [[2, 1]]], 3);
    const b = matrix([[[0, 1]], [[1, 1]], [[2, 1]]], 3);
    const ch = compareAttentionChannels({ a, b, align: [0, 1, 2], created: [] }, -1, edit)!;
    expect(ch.rowsChanged).toBe(0);
    expect(ch.redistributedMass).toBeLessThan(0.02);
    expect(ch.nonLocalRows).toBe(0);
    expect(classifyScope(ch, edit).kind).toBe("none");
  });

  it("reports the shift as a share of the whole layer", () => {
    const a = matrix([[[0, 1]], [[0, 1]]], 2);
    const b = matrix([[[1, 1]], [[1, 1]]], 2);
    const ch = compareAttentionChannels({ a, b, align: [0, 1], created: [] }, -1, edit)!;
    // both rows moved their entire mass to another comparable key
    expect(ch.redistributedMass).toBeCloseTo(1, 6);
    expect(ch.rowMaxTv).toBeCloseTo(1, 6);
    expect(ch.rowsChanged).toBe(2);
    expect(ch.nonLocalShare).toBeCloseTo(1, 6);
    expect(ch.sameSize).toBe(true);
  });

  it("refuses pairs with nothing comparable", () => {
    const a = matrix([[[0, 1]]], 1);
    const b = matrix([[[0, 1]], [[1, 1]]], 2);
    expect(compareAttentionChannels({ a, b, align: [], created: [0, 1] }, -1, edit)).toBeNull();
  });
});

describe("buildVerdict", () => {
  const texts: Record<number, string> = { 1: "ab", 2: " ", 3: "c", 4: "a", 5: "b" };
  const pA = [1, 2, 3];
  const pB = [4, 2, 5, 2, 3];

  /** a square whose row-mean key mass is exactly `diag` */
  const square = (diag: number[]) => {
    const n = diag.length;
    const data = new Float32Array(n * n);
    for (let i = 0; i < n; i++) data[i * n + i] = diag[i] * n;
    return { heads: 1, n, data };
  };

  function base(over: Partial<Parameters<typeof buildVerdict>[0]> = {}) {
    const edit = editSummary("ab c", "a b c", pA, pB, texts)!;
    // the insert splits a's single space token into two on b's side, so the new
    // token walks away with part of the mass that used to sit on a's token 1.
    // everything else is deliberately flat: the verdict should read "basically
    // unchanged" for a position that merely shed a share of its own mass.
    const massA = new Float32Array([0.1, 0.3, 0.14]);
    const massB = new Float32Array([0.1, 0.05, 0.1, 0.1425, 0.1425]);
    void square;
    return {
      promptA: pA,
      promptB: pB,
      texts,
      tagA: "a",
      tagB: "b",
      lensLabel: "the lens",
      measured: "L0 / 3 · mean head",
      edit,
      channels: null,
      nextTop: { idA: 3, idB: 3, pA: 0.44, pB: 0.4003, js: 0.0004, tv: 0.001, sameTop: true },
      attention: {
        head: -1,
        massA,
        massB,
        stats: { maxAbs: 0.002, at: { i: 1, j: 1 }, meanAbs: 0.0004, maxA: 0.5, maxB: 0.5 },
        peakA: 0.5,
        peakB: 0.5,
        selfA: 0.3,
        selfB: 0.3,
      },
      resid: { layers: 3, maxAbs: 0.4, at: 1, meanAbs: 0.2 },
      residScale: 40,
      lens: { diverge: null, agree: 3, layers: 3, firstKeyShift: null, lastKeyShift: null },
      ...over,
    };
  }

  it("calls a one-whitespace insert with no side effects basically unchanged", () => {
    const v = buildVerdict(base());
    expect(v.severity.level).toBe("minimal");
    expect(v.severity.label).toBe("basically unchanged");
    expect(v.change).toContain("+1 chars");
    // the edit itself is always reported first, named by the token it lands on
    expect(v.observations[0].kind).toBe("edit");
    expect(v.observations[0].text).toContain("inserted");
    expect(v.observations[0].text).toContain("token 1");
    // the closing line stays short: one sentence, no numbers, and it does not
    // re-list the evidence the bullets carry
    expect(v.conclusion).toBe("the model reads this sentence the same way");
    expect(v.conclusion.split(/\s+/).length).toBeLessThan(12);
    // ...and the answer is stated before the evidence, in both renderings
    expect(v.text.startsWith(v.answer)).toBe(true);
    expect(v.text).toContain("evidence:");
    expect(v.text.split("\n").length).toBeGreaterThan(4);
  });

  it("answers the global-perturbation question on the first line", () => {
    const v = buildVerdict(base());
    expect(v.answer).toContain("Does one character perturb the layer globally?");
    expect(v.answer).toContain("No");
    // it names the slice it measured, so a single layer is never read as the model
    expect(v.answer).toContain("(L0 / 3 · mean head)");
    expect(v.answer.split(/\s+/).length).toBeLessThan(24);
  });

  it("flags a split token that walks off with real attention", () => {
    const v = buildVerdict(
      base({
        channels: {
          redistributedMass: 0.04,
          rowsChanged: 3,
          rowsTotal: 15,
          rowMaxTv: 0.11,
          rowMaxAt: 4,
          localShare: 0.45,
          nonLocalShare: 0.55,
          nonLocalRows: 2,
          sameSize: true,
        },
        attention: {
          head: -1,
          massA: new Float32Array([0.1, 0.3, 0.14]),
          massB: new Float32Array([0.1, 0.2, 0.05, 0.145, 0.145]),
          stats: { maxAbs: 0.002, at: { i: 1, j: 1 }, meanAbs: 0.0004, maxA: 0.5, maxB: 0.5 },
          peakA: 0.5,
          peakB: 0.5,
          selfA: 0.3,
          selfB: 0.3,
        },
      }),
    );
    expect(v.observations.some((o) => o.text.includes("of the layer's attention mass"))).toBe(true);
    expect(v.severity.level).not.toBe("minimal");
    // the first line answers the reach question; the numbers stay in the evidence
    expect(v.answer).toContain("globally? Yes");
    expect(v.answer).toContain("the whole layer is re-drawn");
    expect(v.text.startsWith(v.answer)).toBe(true);
  });

  it("puts the scope on the clipboard and in the verdict", () => {
    const v = buildVerdict(base());
    expect(v.scope.kind).toBe("none");
    expect(v.scope.text).toContain("no attention pair");
    expect(v.text).toContain("scope:");
  });

  it("keeps a re-split inside the edited span labelled local", () => {
    const v = buildVerdict(
      base({
        channels: {
          redistributedMass: 0.03,
          rowsChanged: 2,
          rowsTotal: 15,
          rowMaxTv: 0.09,
          rowMaxAt: 1,
          localShare: 0.9,
          nonLocalShare: 0.1,
          nonLocalRows: 1,
          sameSize: true,
        },
      }),
    );
    expect(v.scope.kind).toBe("local");
    expect(v.answer).toContain("globally? No");
    expect(v.answer).toContain("inside the edit's own token");
    expect(v.conclusion).toBe("the model's next move is unchanged");
  });

  it("calls a perturbation that re-draws most of the layer global", () => {
    const v = buildVerdict(
      base({
        channels: {
          redistributedMass: 0.2,
          rowsChanged: 12,
          rowsTotal: 15,
          rowMaxTv: 0.7,
          rowMaxAt: 4,
          localShare: 0.2,
          nonLocalShare: 0.8,
          nonLocalRows: 9,
          sameSize: true,
        },
      }),
    );
    expect(v.scope.kind).toBe("global");
    expect(v.answer).toContain("globally? Yes");
    expect(v.conclusion).toBe("the model's next move is unchanged");
  });

  it("calls a flipped next token and moved mass a real change", () => {
    const v = buildVerdict(
      base({
        nextTop: { idA: 3, idB: 5, pA: 0.5, pB: 0.2, js: 0.4, tv: 0.6, sameTop: false },
        attention: {
          head: -1,
          massA: new Float32Array([0.1, 0.2, 0.6]),
          massB: new Float32Array([0.1, 0.06, 0.04, 0.15, 0.25]),
          stats: { maxAbs: 0.42, at: { i: 2, j: 1 }, meanAbs: 0.02, maxA: 0.9, maxB: 0.6 },
          peakA: 0.9,
          peakB: 0.6,
          selfA: 0.3,
          selfB: 0.45,
        },
        lens: { diverge: 1, agree: 1, layers: 3, firstKeyShift: 1, lastKeyShift: 1 },
        channels: {
          redistributedMass: 0.31,
          rowsChanged: 6,
          rowsTotal: 15,
          rowMaxTv: 0.62,
          rowMaxAt: 2,
          localShare: 0.2,
          nonLocalShare: 0.8,
          nonLocalRows: 5,
          sameSize: true,
        },
      }),
    );
    expect(v.severity.level).toBe("major");
    expect(v.severity.label).toBe("material change");
    expect(v.observations.some((o) => o.text.includes("flips"))).toBe(true);
    expect(
      v.observations.some((o) => o.level === "high" && o.text.includes("largest move")),
    ).toBe(true);
    // the first line names what moved: the displacement, the lens, the pick
    expect(v.answer).toContain("globally? Yes");
    expect(v.conclusion).toContain("next-token choice flips");
    expect(v.conclusion).toContain("lens starts splitting at L0");
  });

  it("still answers when attention is missing", () => {
    const v = buildVerdict(base({ attention: null }));
    expect(v.observations.some((o) => o.text.includes("no attention matrices"))).toBe(true);
    expect(v.conclusion.length).toBeGreaterThan(10);
  });

  it("says the lens agreed when it did", () => {
    const v = buildVerdict(base());
    expect(v.observations.some((o) => o.text.includes("all 3 layers agree"))).toBe(true);
  });

  it("tells a re-converged stack from one that never comes back", () => {
    const v = buildVerdict(
      base({ lens: { diverge: 1, agree: 2, layers: 5, firstKeyShift: 1, lastKeyShift: 1 } }),
    );
    expect(v.observations.some((o) => o.text.includes("re-converges"))).toBe(true);
    const never = buildVerdict(
      base({ lens: { diverge: 0, agree: 1, layers: 5, firstKeyShift: 0, lastKeyShift: 4 } }),
    );
    expect(never.observations.some((o) => o.text.includes("never reconverges"))).toBe(true);
  });

  it("never claims attention moved when it did not", () => {
    // attention flat, but the next-token distribution torn apart: the verdict
    // has to lead with the distribution, not invent an attention move
    const v = buildVerdict(
      base({
        nextTop: { idA: 3, idB: 5, pA: 0.5, pB: 0.2, js: 0.49, tv: 0.72, sameTop: false },
        lens: { diverge: 0, agree: 0, layers: 3, firstKeyShift: 0, lastKeyShift: 0 },
        attention: {
          head: -1,
          massA: new Float32Array([0.1, 0.3, 0.14]),
          massB: new Float32Array([0.1, 0.05, 0.14, 0.14, 0.14]),
          stats: { maxAbs: 0.003, at: { i: 1, j: 1 }, meanAbs: 0.0004, maxA: 0.5, maxB: 0.5 },
          peakA: 0.5,
          peakB: 0.5,
          selfA: 0.3,
          selfB: 0.3,
        },
      }),
    );
    expect(v.conclusion).not.toContain("attention moves");
    expect(v.conclusion).toContain("next-token choice flips");
    expect(v.conclusion.split(/\s+/).length).toBeLessThan(45);
  });
});
