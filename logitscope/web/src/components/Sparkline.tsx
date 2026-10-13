// entropy across the whole sequence — the scope trace. spikes are the places
// the model hesitated; hover to inspect them in every other view.

import { entropyColor, useCanvas, type Ctx2D } from "../canvas";
import { focusNode, store, useStore } from "../store";

const PAD_X = 10;
const PAD_Y = 7;

function xFor(i: number, n: number, w: number): number {
  if (n <= 1) return PAD_X;
  return PAD_X + (i / (n - 1)) * (w - 2 * PAD_X);
}

export default function Sparkline() {
  const path = useStore((s) => s.path);
  const nodes = useStore((s) => s.nodes);
  const focus = useStore((s) => focusNode(s));

  const ref = useCanvas(
    (c: Ctx2D) => {
      const { ctx, w, h } = c;
      ctx.clearRect(0, 0, w, h);
      const ents = path.map((id) => nodes[id]?.entropy ?? 0);
      if (ents.length === 0) return;
      const hMax = Math.max(2.5, ...ents);
      const yFor = (e: number) => h - PAD_Y - (e / hMax) * (h - 2 * PAD_Y);

      // faint gridline at 1 nat steps
      ctx.strokeStyle = "rgba(77,255,166,0.08)";
      ctx.lineWidth = 1;
      for (let g = 1; g <= hMax; g++) {
        ctx.beginPath();
        ctx.moveTo(PAD_X, yFor(g));
        ctx.lineTo(w - PAD_X, yFor(g));
        ctx.stroke();
      }

      // the trace
      ctx.strokeStyle = "rgba(77,255,166,0.7)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ents.forEach((e, i) => {
        const x = xFor(i, ents.length, w);
        const y = yFor(e);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // dots, colored like the chips
      ents.forEach((e, i) => {
        const id = path[i];
        ctx.fillStyle = entropyColor(e);
        const r = id === focus ? 3.5 : 2;
        ctx.beginPath();
        ctx.arc(xFor(i, ents.length, w), yFor(e), r, 0, Math.PI * 2);
        ctx.fill();
      });

      // focus marker
      if (focus !== null) {
        const i = path.indexOf(focus);
        if (i >= 0) {
          const x = xFor(i, ents.length, w);
          ctx.strokeStyle = "rgba(201,228,212,0.35)";
          ctx.beginPath();
          ctx.moveTo(x, PAD_Y);
          ctx.lineTo(x, h - PAD_Y);
          ctx.stroke();
        }
      }
    },
    [path, nodes, focus],
  );

  const indexAt = (ev: React.MouseEvent<HTMLCanvasElement>): number | null => {
    const el = ev.currentTarget;
    const rect = el.getBoundingClientRect();
    const n = path.length;
    if (n === 0) return null;
    const t = (ev.clientX - rect.left - PAD_X) / Math.max(1, rect.width - 2 * PAD_X);
    return Math.max(0, Math.min(n - 1, Math.round(t * (n - 1))));
  };

  return (
    <div className="spark panel">
      <div className="panel-title">entropy</div>
      <canvas
        ref={ref}
        className="spark-canvas"
        onMouseMove={(ev) => {
          const i = indexAt(ev);
          store.hover(i === null ? null : path[i]);
        }}
        onMouseLeave={() => store.hover(null)}
        onClick={(ev) => {
          const i = indexAt(ev);
          if (i !== null) store.select(path[i]);
        }}
      />
    </div>
  );
}
