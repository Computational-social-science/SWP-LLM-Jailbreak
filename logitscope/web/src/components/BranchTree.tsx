// every leaf of the session tree is a timeline. the rail lists them, shows
// where each split off from the path you're on, and click = time travel.
// hidden until a fork exists — most sessions never need it.

import { branchTips, pathTo, store, useStore } from "../store";
import { visible } from "../tokens";

export default function BranchTree() {
  const nodes = useStore((s) => s.nodes);
  const tip = useStore((s) => s.tip);
  const path = useStore((s) => s.path);
  const generating = useStore((s) => s.generating);

  const tips = branchTips(nodes);
  if (tips.length < 2) return null;

  return (
    <div className="branch-rail panel">
      <div className="panel-title">branches</div>
      <div className="branch-list">
        {tips.map((t) => {
          const p = pathTo(nodes, t);
          let k = 0; // where this branch leaves the current path
          while (k < p.length && k < path.length && p[k] === path[k]) k++;
          const current = t === tip;
          const preview = p
            .slice(k, k + 4)
            .map((id) => nodes[id]?.text ?? "")
            .join("");
          return (
            <button
              key={t}
              className={current ? "branch branch-on" : "branch"}
              disabled={generating}
              onClick={() => store.checkout(t)}
            >
              <span className="branch-head">
                {current ? "●" : "⑂"} #{t} · {p.length} tok
              </span>
              <span className="branch-preview">
                {current ? "current timeline" : `@${k} ${visible(preview)}…`}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
