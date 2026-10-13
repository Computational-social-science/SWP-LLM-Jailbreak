// the text as the model grew it: one chip per token, colored by how sure the
// model was when it picked it. prompt tokens are chips too (dim ones) so the
// attention ribbons have real targets to land on.

import { useEffect, useRef } from "react";
import { entropyColor } from "../canvas";
import { store, useStore } from "../store";
import RibbonOverlay from "./RibbonOverlay";

function chipTint(entropy: number | null): { background: string; borderBottomColor: string } {
  const c = entropyColor(entropy ?? 0);
  // rgb(...) -> rgba with a whisper of alpha for the fill
  const fill = c.replace("rgb(", "rgba(").replace(")", ",0.13)");
  return { background: fill, borderBottomColor: c };
}

export default function StreamView() {
  const session = useStore((s) => s.session);
  const nodes = useStore((s) => s.nodes);
  const path = useStore((s) => s.path);
  const selected = useStore((s) => s.selected);
  const hovered = useStore((s) => s.hovered);
  const hoverTokenId = useStore((s) => s.hoverTokenId);
  const generating = useStore((s) => s.generating);
  const texts = useStore((s) => s.tokenTexts);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [path.length]);

  if (!session)
    return (
      <div className="stream-wrap panel">
        <div className="panel-title">stream</div>
        <div className="placeholder">type a prompt below to start a session</div>
      </div>
    );

  const promptLen = session.promptIds.length;

  return (
    <div className="stream-wrap panel">
      <div className="panel-title">
        stream — {path.length} tokens{generating ? " ▸ generating" : ""}
      </div>
      <div className="stream-clip">
        <div className="stream" ref={scrollRef}>
          {session.promptIds.map((tid, p) => (
            <span
              key={`p${p}`}
              className="chip chip-prompt"
              data-pos={p}
              onMouseEnter={() => store.set({ hoverPos: p })}
              onMouseLeave={() => store.set({ hoverPos: null })}
            >
              {texts[tid] ?? "·"}
            </span>
          ))}
          {path.map((id, idx) => {
            const n = nodes[id];
            const cls = [
              "chip",
              id === selected ? "chip-selected" : "",
              id === hovered ? "chip-hovered" : "",
              hoverTokenId !== null && n.token_id === hoverTokenId ? "chip-lens-hit" : "",
            ]
              .filter(Boolean)
              .join(" ");
            return (
              <span
                key={id}
                className={cls}
                style={chipTint(n.entropy)}
                data-pos={promptLen + idx}
                title={`#${id} rank ${n.rank ?? "?"}  logprob ${n.logprob?.toFixed(3) ?? "?"}  H ${n.entropy?.toFixed(2) ?? "?"} nats`}
                onClick={() => store.select(id === selected ? null : id)}
                onMouseEnter={() => {
                  store.hover(id);
                  store.set({ hoverPos: promptLen + idx });
                }}
                onMouseLeave={() => {
                  store.hover(null);
                  store.set({ hoverPos: null });
                }}
              >
                {n.text}
                {n.rank !== null && n.rank > 1 && <sup className="chip-rank">{n.rank}</sup>}
              </span>
            );
          })}
          {generating && <span className="cursor">▮</span>}
        </div>
        <RibbonOverlay streamRef={scrollRef} />
      </div>
    </div>
  );
}
