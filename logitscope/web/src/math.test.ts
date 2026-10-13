import { describe, expect, it } from "vitest";
import { crystallizedAt, entropyNats, softmax, topkTrueProbs } from "./math";

describe("softmax", () => {
  it("sums to one and keeps the order", () => {
    const p = softmax([1, 3, 2]);
    expect(p[0] + p[1] + p[2]).toBeCloseTo(1, 6);
    expect(p[1]).toBeGreaterThan(p[2]);
    expect(p[2]).toBeGreaterThan(p[0]);
  });

  it("is shift invariant (and doesn't overflow on big logits)", () => {
    const a = softmax([1, 2, 3]);
    const b = softmax([1001, 1002, 1003]);
    for (let i = 0; i < 3; i++) expect(b[i]).toBeCloseTo(a[i], 6);
  });
});

describe("entropyNats", () => {
  it("uniform is ln(n)", () => {
    expect(entropyNats([0.25, 0.25, 0.25, 0.25])).toBeCloseTo(Math.log(4), 6);
  });

  it("one-hot is zero, and p=0 terms don't nan out", () => {
    expect(entropyNats([1, 0, 0])).toBe(0);
  });
});

describe("topkTrueProbs", () => {
  // ground truth: a small full vocab we can softmax exactly
  const vocab = [4.0, 3.1, 2.9, 1.0, 0.5, -0.3, -1.2, -5.0];
  const full = softmax(vocab);
  const k = 4;
  const ids = [0, 1, 2, 3]; // top-4 by construction
  const logits = vocab.slice(0, k);

  it("recovers exact full-vocab probabilities from the chosen logprob", () => {
    const chosen = 1;
    const { probs, exact } = topkTrueProbs(logits, ids, chosen, Math.log(full[chosen]));
    expect(exact).toBe(true);
    for (let i = 0; i < k; i++) expect(probs[i]).toBeCloseTo(full[i], 6);
  });

  it("plain topk softmax would have overstated them (why the trick matters)", () => {
    const naive = softmax(logits);
    for (let i = 0; i < k; i++) expect(naive[i]).toBeGreaterThan(full[i]);
  });

  it("falls back to topk softmax when the chosen token isn't in the list", () => {
    const { probs, exact } = topkTrueProbs(logits, ids, 7, Math.log(full[7]));
    expect(exact).toBe(false);
    let sum = 0;
    for (const p of probs) sum += p;
    expect(sum).toBeCloseTo(1, 6);
  });
});

describe("crystallizedAt", () => {
  const from = (tops: number[]) => (l: number) => tops[l];

  it("finds the layer where the run to the top starts", () => {
    expect(crystallizedAt(from([9, 9, 5, 5, 5]), 5, 5)).toBe(2);
  });

  it("a flicker mid-stack doesn't count", () => {
    // agrees at L1, disagrees at L2, agrees again above
    expect(crystallizedAt(from([9, 5, 9, 5, 5]), 5, 5)).toBe(3);
  });

  it("null when even the final layer disagrees (rank>1 pick)", () => {
    expect(crystallizedAt(from([1, 2, 3]), 3, 5)).toBeNull();
  });

  it("crystallized from the embeddings themselves", () => {
    expect(crystallizedAt(from([5, 5, 5]), 3, 5)).toBe(0);
  });
});
