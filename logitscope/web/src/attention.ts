// attention-matrix geometry + drawing, shared by the single attention view and
// the two-column compare view. the heatmap is drawn into an n x n offscreen
// bitmap and scaled up with smoothing off so it stays crisp at any context
// length.

import { attnRGB, type Ctx2D } from "./canvas";
import type { AttnMatrix } from "./store";

const PAD = 8;

/** weight at (query i, key j) for a head, or the mean over heads. */
export function attnAt(m: AttnMatrix, head: number, i: number, j: number): number {
  if (head >= 0) return m.data[head * m.n * m.n + i * m.n + j];
  let sum = 0;
  for (let h = 0; h < m.heads; h++) sum += m.data[h * m.n * m.n + i * m.n + j];
  return sum / m.heads;
}

export function drawAttention(
  c: Ctx2D,
  matrix: AttnMatrix,
  head: number,
  promptLen: number,
  hover: { i: number; j: number } | null,
): void {
  const { ctx, w, h } = c;
  ctx.clearRect(0, 0, w, h);
  const n = matrix.n;
  const size = Math.min(w, h) - 2 * PAD;
  if (size <= 0 || n === 0) return;

  const off = document.createElement("canvas");
  off.width = n;
  off.height = n;
  const octx = off.getContext("2d")!;
  const img = octx.createImageData(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      const [r, g, b] = attnRGB(attnAt(matrix, head, i, j));
      const o = (i * n + j) * 4;
      img.data[o] = r;
      img.data[o + 1] = g;
      img.data[o + 2] = b;
      img.data[o + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);

  const x0 = PAD;
  const y0 = PAD;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, x0, y0, size, size);

  // where the prompt ends
  const cell = size / n;
  if (promptLen > 0 && promptLen < n) {
    ctx.strokeStyle = "rgba(255,200,87,0.5)";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x0 + promptLen * cell, y0);
    ctx.lineTo(x0 + promptLen * cell, y0 + size);
    ctx.moveTo(x0, y0 + promptLen * cell);
    ctx.lineTo(x0 + size, y0 + promptLen * cell);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  if (hover && hover.i < n && hover.j < n) {
    ctx.strokeStyle = "rgba(201,228,212,0.6)";
    ctx.strokeRect(x0 + hover.j * cell, y0 + hover.i * cell, cell, cell);
    ctx.strokeStyle = "rgba(201,228,212,0.15)";
    ctx.strokeRect(x0, y0 + hover.i * cell, size, cell);
    ctx.strokeRect(x0 + hover.j * cell, y0, cell, size);
  }
}

/** canvas-pixel -> matrix cell, lower-triangular by construction. */
export function cellFromMouse(
  ev: { clientX: number; clientY: number; currentTarget: HTMLCanvasElement },
  matrix: AttnMatrix,
): { i: number; j: number } | null {
  const rect = ev.currentTarget.getBoundingClientRect();
  const size = Math.min(rect.width, rect.height) - 2 * PAD;
  const j = Math.floor(((ev.clientX - rect.left - PAD) / size) * matrix.n);
  const i = Math.floor(((ev.clientY - rect.top - PAD) / size) * matrix.n);
  if (i < 0 || j < 0 || i >= matrix.n || j > i) return null;
  return { i, j };
}
