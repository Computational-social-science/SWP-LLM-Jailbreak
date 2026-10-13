// token-level diff between two branches. plain lcs dp — branches share their
// fork prefix by construction (and often re-converge at the end), so trim the
// common ends first and only dp the disagreement in the middle.

export type DiffOp =
  | { kind: "same"; a: number; b: number } // indices into a and b
  | { kind: "del"; a: number } // only in a
  | { kind: "ins"; b: number }; // only in b

export function diffTokens(a: ArrayLike<number>, b: ArrayLike<number>): DiffOp[] {
  let lo = 0;
  while (lo < a.length && lo < b.length && a[lo] === b[lo]) lo++;
  let aHi = a.length;
  let bHi = b.length;
  while (aHi > lo && bHi > lo && a[aHi - 1] === b[bHi - 1]) {
    aHi--;
    bHi--;
  }

  const n = aHi - lo;
  const m = bHi - lo;
  // dp[i][j] = lcs length of a[lo..lo+i) vs b[lo..lo+j). typed-array rows keep
  // this cheap; branches are a few hundred tokens at most.
  const dp: Uint32Array[] = [];
  for (let i = 0; i <= n; i++) dp.push(new Uint32Array(m + 1));
  for (let i = 1; i <= n; i++)
    for (let j = 1; j <= m; j++)
      dp[i][j] =
        a[lo + i - 1] === b[lo + j - 1]
          ? dp[i - 1][j - 1] + 1
          : Math.max(dp[i - 1][j], dp[i][j - 1]);

  const mid: DiffOp[] = [];
  let i = n;
  let j = m;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && a[lo + i - 1] === b[lo + j - 1]) {
      mid.push({ kind: "same", a: lo + i - 1, b: lo + j - 1 });
      i--;
      j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      mid.push({ kind: "ins", b: lo + j - 1 });
      j--;
    } else {
      mid.push({ kind: "del", a: lo + i - 1 });
      i--;
    }
  }
  mid.reverse();

  const out: DiffOp[] = [];
  for (let k = 0; k < lo; k++) out.push({ kind: "same", a: k, b: k });
  out.push(...mid);
  for (let k = 0; aHi + k < a.length; k++) out.push({ kind: "same", a: aHi + k, b: bHi + k });
  return out;
}
