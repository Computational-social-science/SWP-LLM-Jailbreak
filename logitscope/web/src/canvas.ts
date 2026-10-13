// canvas plumbing shared by every view: dpr-crisp sizing, redraw-on-resize,
// and the color scales.

import { useEffect, useRef } from "react";
import { clamp01 } from "./math";

export interface Ctx2D {
  ctx: CanvasRenderingContext2D;
  w: number; // css pixels — draw in these, the transform handles dpr
  h: number;
}

export function fitCanvas(canvas: HTMLCanvasElement): Ctx2D | null {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (w === 0 || h === 0) return null;
  const bw = Math.round(w * dpr);
  const bh = Math.round(h * dpr);
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

/** a canvas that redraws when deps change or the element resizes. */
export function useCanvas(
  draw: (c: Ctx2D) => void,
  deps: unknown[],
): React.RefObject<HTMLCanvasElement | null> {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const redraw = () => {
      const c = fitCanvas(canvas);
      if (c) draw(c);
    };
    redraw();
    const ro = new ResizeObserver(redraw);
    ro.observe(canvas);
    return () => ro.disconnect();
    // eslint-style exhaustive-deps doesn't apply: deps are the caller's data
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps
  return ref;
}

// -- color scales --

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function mix(c0: [number, number, number], c1: [number, number, number], t: number): string {
  return `rgb(${Math.round(lerp(c0[0], c1[0], t))},${Math.round(lerp(c0[1], c1[1], t))},${Math.round(lerp(c0[2], c1[2], t))})`;
}

/** blend two rgb triples. exported because the compare view needs signed ramps
 * (its delta heatmaps go red one way, phosphor the other). */
export function mixRGB(
  c0: [number, number, number],
  c1: [number, number, number],
  t: number,
): [number, number, number] {
  return [
    Math.round(lerp(c0[0], c1[0], t)),
    Math.round(lerp(c0[1], c1[1], t)),
    Math.round(lerp(c0[2], c1[2], t)),
  ];
}

const PHOS: [number, number, number] = [77, 255, 166];
const AMBER: [number, number, number] = [255, 200, 87];
const RED: [number, number, number] = [255, 93, 93];

/** confident -> phosphor green, unsure -> amber, lost -> red. ~3 nats = lost. */
export function entropyColor(entropy: number): string {
  const x = clamp01(entropy / 3);
  return x < 0.5 ? mix(PHOS, AMBER, x * 2) : mix(AMBER, RED, (x - 0.5) * 2);
}

/** probability -> phosphor tint with alpha, for cell shading on dark bg. */
export function probShade(p: number): string {
  const a = 0.06 + 0.8 * clamp01(p);
  return `rgba(77,255,166,${a.toFixed(3)})`;
}

/** attention weight -> green ramp; gamma lift so small weights stay visible. */
export function attnRGB(w: number): [number, number, number] {
  const x = clamp01(w) ** 0.45;
  return [Math.round(8 + 60 * x), Math.round(16 + 220 * x), Math.round(12 + 140 * x)];
}

export function attnShade(w: number): string {
  const [r, g, b] = attnRGB(w);
  return `rgb(${r},${g},${b})`;
}

/** signed ramp for "how much did this change" heatmaps: x in [-1, 1], red for
 * one direction and phosphor for the other, gamma-lifted so small deltas stay
 * visible on a near-black background. */
export function deltaRGB(x: number): [number, number, number] {
  const v = Math.max(-1, Math.min(1, x));
  const m = Math.abs(v) ** 0.45;
  const base: [number, number, number] = [10, 16, 13];
  const hot: [number, number, number] = v >= 0 ? PHOS : RED;
  return mixRGB(base, hot, m);
}

export function deltaShade(x: number, alpha = 0.85): string {
  const [r, g, b] = deltaRGB(x);
  return `rgba(${r},${g},${b},${alpha})`;
}
