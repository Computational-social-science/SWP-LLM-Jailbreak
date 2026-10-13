// what it almost said: the top-k candidates at the focused step, with their
// exact probabilities (see topkTrueProbs), the pick highlighted. clicking a
// candidate forks the timeline right there — the fun part of the whole tool.

import { useEffect, useState } from "react";
import { forkAt } from "../actions";
import { entropyColor, useCanvas, type Ctx2D } from "../canvas";
import { topkTrueProbs } from "../math";
import { focusNode, useStore } from "../store";
import { requestTexts, visible } from "../tokens";

const ROWS = 12;
const ROW_H = 19;

export default function DetailPanel() {
  const focus = useStore((s) => focusNode(s));
  const node = useStore((s) => (focus !== null ? s.nodes[focus] : undefined));
  const step = useStore((s) => (focus !== null ? s.steps[focus] : undefined));
  const texts = useStore((s) => s.tokenTexts);
  const generating = useStore((s) => s.generating);
  const [row, setRow] = useState<number | null>(null);

  useEffect(() => {
    if (step?.topkIds) requestTexts(step.topkIds);
  }, [step]);

  const ref = useCanvas(
    (c: Ctx2D) => {
      const { ctx, w, h } = c;
      ctx.clearRect(0, 0, w, h);
      if (!node || !step?.topkIds || !step.topkLogits || node.token_id === null) return;
      const { probs, exact } = topkTrueProbs(
        step.topkLogits,
        step.topkIds,
        node.token_id,
        node.logprob ?? 0,
      );
      const n = Math.min(ROWS, step.topkIds.length);
      const pMax = probs.reduce((a, b) => Math.max(a, b), 1e-9);
      const labelW = 92;
      const pctW = 52;
      const barW = w - labelW - pctW - 16;
      ctx.font = "11px ui-monospace, Menlo, monospace";
      ctx.textBaseline = "middle";
      for (let i = 0; i < n; i++) {
        const y = 6 + i * ROW_H + ROW_H / 2;
        const id = step.topkIds[i];
        const isPick = id === node.token_id;
        const p = probs[i];

        if (i === row && !isPick && !generating) {
          ctx.fillStyle = "rgba(255,200,87,0.09)";
          ctx.fillRect(2, y - ROW_H / 2 + 1, w - 4, ROW_H - 2);
        }

        ctx.fillStyle = isPick ? "#4dffa6" : "#6d8a7b";
        const label = texts[id] !== undefined ? visible(texts[id]) : `#${id}`;
        ctx.fillText(label.slice(0, 11), 8, y, labelW - 4);

        const bw = Math.max(1, (p / pMax) * barW);
        ctx.fillStyle = isPick ? "rgba(77,255,166,0.85)" : "rgba(77,255,166,0.28)";
        ctx.fillRect(labelW + 8, y - 5, bw, 10);
        if (isPick) {
          ctx.strokeStyle = "#4dffa6";
          ctx.strokeRect(labelW + 7.5, y - 5.5, barW + 1, 11);
        }

        ctx.fillStyle = isPick ? "#c9e4d4" : "#6d8a7b";
        const pct = p >= 0.001 ? `${(p * 100).toFixed(1)}%` : `${(p * 100).toExponential(0)}%`;
        ctx.fillText(`${exact ? "" : "~"}${pct}`, labelW + 12 + barW, y);
      }
    },
    [node, step, texts, row, generating],
  );

  const rowAt = (ev: React.MouseEvent<HTMLCanvasElement>): number | null => {
    if (!step?.topkIds) return null;
    const rect = ev.currentTarget.getBoundingClientRect();
    const i = Math.floor((ev.clientY - rect.top - 6) / ROW_H);
    return i >= 0 && i < Math.min(ROWS, step.topkIds.length) ? i : null;
  };

  return (
    <div className="detail-wrap panel">
      <div className="panel-title">token detail — click a bar to fork</div>
      {node && node.token_id !== null ? (
        <>
          <div className="detail-head">
            <span className="detail-token" style={{ color: entropyColor(node.entropy ?? 0) }}>
              {visible(node.text)}
            </span>
            <span className="detail-stats">
              id {node.token_id} · rank {node.rank ?? "?"} · p{" "}
              {node.logprob !== null ? `${(Math.exp(node.logprob) * 100).toFixed(1)}%` : "?"} · H{" "}
              {node.entropy?.toFixed(2) ?? "?"}
            </span>
          </div>
          <canvas
            ref={ref}
            className="detail-canvas"
            style={{ height: ROWS * ROW_H + 12, cursor: generating ? "default" : "pointer" }}
            onMouseMove={(ev) => setRow(rowAt(ev))}
            onMouseLeave={() => setRow(null)}
            onClick={(ev) => {
              const i = rowAt(ev);
              if (i === null || focus === null || !step?.topkIds || generating) return;
              const alt = step.topkIds[i];
              // forking onto the token it already picked is just the same timeline
              if (alt !== node.token_id) forkAt(focus, alt);
            }}
          />
        </>
      ) : (
        <div className="placeholder">hover or click a token</div>
      )}
    </div>
  );
}
