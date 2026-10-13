// the signature view: logit lens per layer. every column is a generated
// token, every row is a layer (embeddings at the bottom, final norm at the
// top), every cell is what the unembedding thinks the next token is at that
// depth. you can watch the answer crystallize on the way up the stack.

import { useEffect, useRef, useState } from "react";
import { probShade, useCanvas, type Ctx2D } from "../canvas";
import { crystallizedAt } from "../math";
import { store, useStore } from "../store";
import { visible } from "../tokens";

const LABEL_W = 46;
const CELL_W = 78;
const HEAD_H = 20;

interface Cell {
  col: number;
  layer: number; // 0 = embeddings
}

export default function Waterfall() {
  const path = useStore((s) => s.path);
  const nodes = useStore((s) => s.nodes);
  const steps = useStore((s) => s.steps);
  const texts = useStore((s) => s.tokenTexts);
  const selected = useStore((s) => s.selected);
  const hovered = useStore((s) => s.hovered);
  const [cell, setCell] = useState<Cell | null>(null);

  // layer count from the first lens we've seen
  let layers = 0;
  for (const id of path) {
    const lens = steps[id]?.lens;
    if (lens) {
      layers = lens.layers;
      break;
    }
  }
  const cellH = layers > 0 ? Math.max(13, Math.min(24, 240 / layers)) : 16;
  const contentW = LABEL_W + path.length * CELL_W + 8;
  const contentH = HEAD_H + layers * cellH + 8;
  const rowY = (l: number) => HEAD_H + (layers - 1 - l) * cellH;

  const scrollRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollLeft = el.scrollWidth;
  }, [path.length]);

  const ref = useCanvas(
    (c: Ctx2D) => {
      const { ctx, w, h } = c;
      ctx.clearRect(0, 0, w, h);
      if (layers === 0) return;
      ctx.font = "10px ui-monospace, Menlo, monospace";
      ctx.textBaseline = "middle";

      // layer labels: row 0 = embeddings, row layers-1 = final
      ctx.fillStyle = "#6d8a7b";
      for (let l = 0; l < layers; l++) {
        const y = rowY(l) + cellH / 2;
        const name = l === 0 ? "emb" : `L${l - 1}`;
        if (cellH >= 13 || l % 2 === 0 || l === layers - 1) ctx.fillText(name, 6, y, LABEL_W - 8);
      }

      path.forEach((id, col) => {
        const x0 = LABEL_W + col * CELL_W;
        const node = nodes[id];
        const lens = steps[id]?.lens;

        if (id === hovered || id === selected) {
          ctx.fillStyle = id === selected ? "rgba(77,255,166,0.10)" : "rgba(201,228,212,0.06)";
          ctx.fillRect(x0, 0, CELL_W, HEAD_H + layers * cellH);
        }

        // header: the token this column actually became
        ctx.fillStyle = id === selected ? "#4dffa6" : "#c9e4d4";
        ctx.fillText(visible(node?.text ?? ""), x0 + 3, HEAD_H / 2, CELL_W - 6);

        if (!lens) return;
        for (let l = 0; l < layers; l++) {
          const y0 = rowY(l);
          const topId = lens.ids[l * lens.k];
          const topP = lens.probs[l * lens.k];
          ctx.fillStyle = probShade(topP);
          ctx.fillRect(x0 + 1, y0 + 1, CELL_W - 2, cellH - 2);
          if (node?.token_id === topId) {
            ctx.strokeStyle = "rgba(77,255,166,0.55)";
            ctx.strokeRect(x0 + 1.5, y0 + 1.5, CELL_W - 3, cellH - 3);
          }
          if (cell && cell.col === col && cell.layer === l) {
            ctx.strokeStyle = "#ffc857";
            ctx.strokeRect(x0 + 0.5, y0 + 0.5, CELL_W - 1, cellH - 1);
          }
          if (cellH >= 13) {
            ctx.fillStyle = topP > 0.4 ? "#0a0e0c" : "#c9e4d4";
            const label = texts[topId] !== undefined ? visible(texts[topId]) : `#${topId}`;
            ctx.fillText(label.slice(0, 9), x0 + 4, y0 + cellH / 2, CELL_W - 8);
          }
        }

        // where the final answer locks in: a solid line under that layer
        if (node?.token_id !== null && node !== undefined) {
          const at = crystallizedAt((l) => lens.ids[l * lens.k], layers, node.token_id!);
          if (at !== null) {
            const y = rowY(at) + cellH - 1;
            ctx.strokeStyle = "#4dffa6";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(x0 + 2, y);
            ctx.lineTo(x0 + CELL_W - 2, y);
            ctx.stroke();
            ctx.lineWidth = 1;
          }
        }
      });
    },
    [path, nodes, steps, texts, layers, cellH, selected, hovered, cell],
  );

  const cellAt = (ev: React.MouseEvent<HTMLCanvasElement>): Cell | null => {
    const rect = ev.currentTarget.getBoundingClientRect();
    const x = ev.clientX - rect.left - LABEL_W;
    const y = ev.clientY - rect.top - HEAD_H;
    const col = Math.floor(x / CELL_W);
    const row = Math.floor(y / cellH);
    if (col < 0 || col >= path.length || row < 0 || row >= layers) return null;
    return { col, layer: layers - 1 - row };
  };

  const inspected = (() => {
    if (!cell) return null;
    const id = path[cell.col];
    const lens = steps[id]?.lens;
    if (!lens) return null;
    const off = cell.layer * lens.k;
    return {
      node: nodes[id],
      layerName: cell.layer === 0 ? "emb" : `L${cell.layer - 1}`,
      rows: Array.from({ length: lens.k }, (_, i) => ({
        id: lens.ids[off + i],
        p: lens.probs[off + i],
      })),
    };
  })();

  return (
    <>
      {layers === 0 ? (
        <div className="placeholder">generate with lens capture on to see the stack think</div>
      ) : (
        <div className="waterfall-body">
          <div className="waterfall-scroll" ref={scrollRef}>
            <canvas
              ref={ref}
              style={{ width: contentW, height: contentH }}
              onMouseMove={(ev) => {
                const c = cellAt(ev);
                setCell(c);
                if (c) {
                  const id = path[c.col];
                  const lens = steps[id]?.lens;
                  store.hover(id);
                  store.hoverToken(lens ? lens.ids[c.layer * lens.k] : null);
                } else {
                  store.hover(null);
                  store.hoverToken(null);
                }
              }}
              onMouseLeave={() => {
                setCell(null);
                store.hover(null);
                store.hoverToken(null);
              }}
              onClick={(ev) => {
                const c = cellAt(ev);
                if (c) store.select(path[c.col]);
              }}
            />
          </div>
          <div className="lens-inspect">
            {inspected ? (
              <>
                <div className="lens-inspect-head">
                  {inspected.layerName} → {visible(inspected.node?.text ?? "")}
                </div>
                {inspected.rows.map((r, i) => (
                  <div key={i} className="lens-row">
                    <span
                      className={
                        r.id === inspected.node?.token_id ? "lens-tok lens-tok-final" : "lens-tok"
                      }
                    >
                      {texts[r.id] !== undefined ? visible(texts[r.id]) : `#${r.id}`}
                    </span>
                    <span className="lens-bar" style={{ width: `${Math.max(2, r.p * 60)}px` }} />
                    <span className="lens-p">{(r.p * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </>
            ) : (
              <div className="lens-inspect-hint">hover a cell</div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
