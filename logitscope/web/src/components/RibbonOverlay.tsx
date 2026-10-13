// attention ribbons: hover any token and see where it looked, drawn straight
// onto the text. weights come from the same square matrix the heatmap uses.

import { useEffect, useRef } from "react";
import { attnAt } from "../attention";
import { useStore } from "../store";

// draw at most this many ribbons, strongest first. an absolute weight cutoff
// looked great on peaky heads and drew nothing at all on flat ones (a uniform
// row over 90 positions never clears any fixed threshold), so everything is
// relative to the row max instead.
const MAX_RIBBONS = 16;

export default function RibbonOverlay({
  streamRef,
}: {
  streamRef: React.RefObject<HTMLDivElement | null>;
}) {
  const hoverPos = useStore((s) => s.hoverPos);
  const matrix = useStore((s) => s.attnMatrix);
  const head = useStore((s) => s.attnHead);
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    const stream = streamRef.current;
    if (!canvas || !stream) return;

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      if (hoverPos === null || !matrix || hoverPos >= matrix.n) return;

      const base = canvas.getBoundingClientRect();
      const anchors = new Map<number, { x: number; yTop: number; yBot: number }>();
      stream.querySelectorAll<HTMLElement>("[data-pos]").forEach((el) => {
        const r = el.getBoundingClientRect();
        anchors.set(Number(el.dataset.pos), {
          x: r.left + r.width / 2 - base.left,
          yTop: r.top - base.top,
          yBot: r.bottom - base.top,
        });
      });

      const from = anchors.get(hoverPos);
      if (!from) return;
      const row: { j: number; w: number }[] = [];
      for (let j = 0; j <= hoverPos; j++) row.push({ j, w: attnAt(matrix, head, hoverPos, j) });
      row.sort((a, b) => b.w - a.w);
      const wMax = row[0]?.w ?? 0;
      if (wMax <= 0) return;

      for (const { j, w } of row.slice(0, MAX_RIBBONS)) {
        const rel = w / wMax;
        if (rel < 0.05) continue;
        const to = anchors.get(j);
        if (!to) continue;
        const lift = Math.min(60, 14 + Math.abs(from.x - to.x) * 0.12);
        ctx.strokeStyle = `rgba(77,255,166,${(0.12 + 0.7 * rel).toFixed(3)})`;
        ctx.lineWidth = 1 + 4 * rel;
        ctx.beginPath();
        ctx.moveTo(from.x, from.yTop);
        ctx.bezierCurveTo(
          from.x,
          from.yTop - lift,
          to.x,
          to.yTop - lift,
          to.x,
          j === hoverPos ? to.yTop : to.yBot - 4,
        );
        ctx.stroke();
      }
    };

    draw();
    stream.addEventListener("scroll", draw, { passive: true });
    const ro = new ResizeObserver(draw);
    ro.observe(canvas);
    return () => {
      stream.removeEventListener("scroll", draw);
      ro.disconnect();
    };
  }, [hoverPos, matrix, head, streamRef]);

  return <canvas ref={ref} className="ribbons" />;
}
