// small numeric helpers for the views. everything works on plain arrays so
// it's trivially testable.

export function softmax(logits: ArrayLike<number>): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

export function entropyNats(probs: ArrayLike<number>): number {
  let h = 0;
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    if (p > 0) h -= p * Math.log(p);
  }
  return h;
}

export interface TopkProbs {
  probs: Float32Array;
  exact: boolean;
}

// softmax over just the top-k logits overstates every probability — the tail
// of the vocab is missing from the denominator. but the server tells us the
// chosen token's TRUE logprob under the full distribution, so if the chosen
// token is in the top-k we can recover the real partition function:
//   log Z = logit(chosen) - logprob(chosen)
// and then p_i = exp(logit_i - log Z) is exact for every candidate.
export function topkTrueProbs(
  logits: ArrayLike<number>,
  ids: ArrayLike<number>,
  chosenId: number,
  chosenLogprob: number,
): TopkProbs {
  let idx = -1;
  for (let i = 0; i < ids.length; i++)
    if (ids[i] === chosenId) {
      idx = i;
      break;
    }
  if (idx < 0) {
    // chosen token fell outside the top-k (wild sampling) — best we can do
    return { probs: softmax(logits), exact: false };
  }
  const logZ = logits[idx] - chosenLogprob;
  const out = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) out[i] = Math.exp(logits[i] - logZ);
  return { probs: out, exact: true };
}

export function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

// lowest layer from which the lens argmax equals the final pick all the way
// up the stack — "where the answer crystallized". flickers (agree, disagree,
// agree again) don't count: the run has to be unbroken to the top.
export function crystallizedAt(
  topIdAt: (layer: number) => number,
  layers: number,
  finalId: number,
): number | null {
  let at: number | null = null;
  for (let l = layers - 1; l >= 0; l--) {
    if (topIdAt(l) === finalId) at = l;
    else break;
  }
  return at;
}
