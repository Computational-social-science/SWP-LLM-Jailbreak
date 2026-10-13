// attention heatmap over the whole context, one layer at a time. rows are
// queries, columns are keys, lower-triangular by construction.

import { useEffect, useState } from "react";
import { fetchAttn } from "../actions";
import { attnAt, cellFromMouse, drawAttention } from "../attention";
import { useCanvas, type Ctx2D } from "../canvas";
import { currentModel, store, useStore } from "../store";
import { visible } from "../tokens";

export default function AttentionView() {
  const session = useStore((s) => s.session);
  const tip = useStore((s) => s.tip);
  const generating = useStore((s) => s.generating);
  const layer = useStore((s) => s.attnLayer);
  const head = useStore((s) => s.attnHead);
  const matrix = useStore((s) => s.attnMatrix);
  const pending = useStore((s) => s.attnPending);
  const model = useStore((s) => currentModel(s));
  const nodes = useStore((s) => s.nodes);
  const path = useStore((s) => s.path);
  const texts = useStore((s) => s.tokenTexts);
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);

  const nLayers = model?.n_layers ?? 0;
  const nHeads = model?.n_heads ?? 0;
  const promptLen = session?.promptIds.length ?? 0;

  // refetch whenever the view is stale: new tokens, new layer, first open
  const stale = !matrix || matrix.layer !== layer || matrix.forNode !== tip;
  useEffect(() => {
    if (session && !generating && stale) fetchAttn();
  }, [session, generating, stale, tip, layer]);

  const posText = (p: number): string => {
    if (p < promptLen) return texts[session?.promptIds[p] ?? -1] ?? `prompt[${p}]`;
    const node = nodes[path[p - promptLen]];
    return node?.text ?? "?";
  };

  const ref = useCanvas(
    (c: Ctx2D) => {
      if (matrix) drawAttention(c, matrix, head, promptLen, hover);
    },
    [matrix, head, hover, promptLen],
  );

  return (
    <div className="attn-body">
      <div className="attn-side">
        <label>
          layer{" "}
          <select
            value={layer}
            onChange={(e) => store.setAttnView(Number(e.target.value), head)}
          >
            {Array.from({ length: nLayers }, (_, l) => (
              <option key={l} value={l}>
                {l}
              </option>
            ))}
          </select>
        </label>
        <label>
          head{" "}
          <select value={head} onChange={(e) => store.setAttnView(layer, Number(e.target.value))}>
            <option value={-1}>mean</option>
            {Array.from({ length: nHeads }, (_, h) => (
              <option key={h} value={h}>
                {h}
              </option>
            ))}
          </select>
        </label>
        <div className="attn-status">
          {pending ? "computing…" : matrix ? `${matrix.n}×${matrix.n}` : "no data"}
        </div>
        {hover && matrix && (
          <div className="attn-readout">
            <div>
              q <span className="attn-tok">{visible(posText(hover.i))}</span>
            </div>
            <div>
              k <span className="attn-tok">{visible(posText(hover.j))}</span>
            </div>
            <div>w {attnAt(matrix, head, hover.i, hover.j).toFixed(4)}</div>
          </div>
        )}
      </div>
      <canvas
        ref={ref}
        className="attn-canvas"
        onMouseMove={(ev) => {
          if (!matrix) return;
          const c = cellFromMouse(ev, matrix);
          setHover(c);
          if (c) {
            const genIdx = c.i - promptLen;
            store.hover(genIdx >= 0 && genIdx < path.length ? path[genIdx] : null);
          }
        }}
        onMouseLeave={() => {
          setHover(null);
          store.hover(null);
        }}
      />
    </div>
  );
}
