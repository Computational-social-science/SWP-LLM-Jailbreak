import { describe, expect, it } from "vitest";
import { diffTokens, type DiffOp } from "./diff";

/** the ops must replay both sequences exactly, in order. */
function checkReplays(a: number[], b: number[], ops: DiffOp[]): void {
  const gotA = ops.filter((o) => o.kind !== "ins").map((o) => a[(o as { a: number }).a]);
  const gotB = ops.filter((o) => o.kind !== "del").map((o) => b[(o as { b: number }).b]);
  expect(gotA).toEqual(a);
  expect(gotB).toEqual(b);
}

describe("diffTokens", () => {
  it("identical sequences are all same", () => {
    const ops = diffTokens([1, 2, 3], [1, 2, 3]);
    expect(ops.every((o) => o.kind === "same")).toBe(true);
    expect(ops).toHaveLength(3);
  });

  it("empty vs empty", () => {
    expect(diffTokens([], [])).toEqual([]);
  });

  it("everything inserted against an empty side", () => {
    const ops = diffTokens([], [7, 8]);
    expect(ops.map((o) => o.kind)).toEqual(["ins", "ins"]);
    checkReplays([], [7, 8], ops);
  });

  it("fork shape: shared prefix, divergent middle, shared tail", () => {
    const a = [1, 2, 3, 4, 9];
    const b = [1, 2, 7, 7, 4, 9];
    const ops = diffTokens(a, b);
    checkReplays(a, b, ops);
    // prefix and tail stay same
    expect(ops[0]).toEqual({ kind: "same", a: 0, b: 0 });
    expect(ops[ops.length - 1]).toEqual({ kind: "same", a: 4, b: 5 });
    expect(ops.filter((o) => o.kind === "same")).toHaveLength(4); // 1 2 4 9
  });

  it("a substitution reads del-then-ins", () => {
    const ops = diffTokens([5], [6]);
    expect(ops.map((o) => o.kind)).toEqual(["del", "ins"]);
  });

  it("nothing in common", () => {
    const a = [1, 2];
    const b = [3, 4, 5];
    const ops = diffTokens(a, b);
    checkReplays(a, b, ops);
    expect(ops.filter((o) => o.kind === "same")).toHaveLength(0);
  });

  it("repeated tokens don't confuse the alignment", () => {
    // "the the cat" vs "the cat the" — lcs is length 2 either way
    const a = [10, 10, 20];
    const b = [10, 20, 10];
    const ops = diffTokens(a, b);
    checkReplays(a, b, ops);
    expect(ops.filter((o) => o.kind === "same").length).toBe(2);
  });

  it("replays hold on messy pseudo-random pairs", () => {
    // deterministic lcg so a failure here is reproducible
    let s = 42;
    const rnd = (n: number) => ((s = (s * 1103515245 + 12345) & 0x7fffffff), s % n);
    for (let round = 0; round < 20; round++) {
      const a = Array.from({ length: rnd(40) }, () => rnd(8));
      const b = Array.from({ length: rnd(40) }, () => rnd(8));
      checkReplays(a, b, diffTokens(a, b));
    }
  });
});
