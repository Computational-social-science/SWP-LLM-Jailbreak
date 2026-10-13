// two timelines side by side with the token-level diff highlighted — what
// actually changed downstream of a fork. row a is the branch you're on, row b
// is any other tip. red = only in a, phosphor = only in b, dim = shared.

import { diffTokens } from "../diff";
import { branchTips, pathTo, store, useStore } from "../store";

export default function CompareView() {
  const nodes = useStore((s) => s.nodes);
  const tip = useStore((s) => s.tip);
  const other = useStore((s) => s.diffOther);
  const generating = useStore((s) => s.generating);

  const tips = branchTips(nodes);
  if (tips.length < 2)
    return <div className="placeholder">fork a token to get a second timeline to diff</div>;

  // default b: the newest tip that isn't the one we're on
  const bTip =
    other !== null && other !== tip && tips.includes(other)
      ? other
      : [...tips].reverse().find((t) => t !== tip)!;
  const pa = pathTo(nodes, tip);
  const pb = pathTo(nodes, bTip);
  const ops = diffTokens(
    pa.map((id) => nodes[id].token_id ?? -1),
    pb.map((id) => nodes[id].token_id ?? -1),
  );

  const chip = (nodeId: number, kind: "same" | "del" | "ins", key: number) => (
    <span
      key={key}
      className={kind === "same" ? "chip chip-ctx" : `chip chip-${kind}`}
      title={`#${nodeId}`}
      onClick={() => {
        if (generating) return;
        // clicking a chip jumps you onto that branch at that token
        store.checkout(kind === "ins" ? bTip : tip);
        store.select(nodeId);
      }}
    >
      {nodes[nodeId]?.text ?? ""}
    </span>
  );

  return (
    <div className="diff-body">
      <div className="diff-row">
        <div className="diff-row-head">
          <span className="diff-tag diff-tag-a">a</span> #{tip} · {pa.length} tok · current
        </div>
        <div className="diff-chips">
          {ops
            .filter((o) => o.kind !== "ins")
            .map((o, i) => chip(pa[(o as { a: number }).a], o.kind as "same" | "del", i))}
        </div>
      </div>
      <div className="diff-row">
        <div className="diff-row-head">
          <span className="diff-tag diff-tag-b">b</span>
          <select
            value={bTip}
            onChange={(e) => store.set({ diffOther: Number(e.target.value) })}
          >
            {tips
              .filter((t) => t !== tip)
              .map((t) => (
                <option key={t} value={t}>
                  #{t} · {pathTo(nodes, t).length} tok
                </option>
              ))}
          </select>
        </div>
        <div className="diff-chips">
          {ops
            .filter((o) => o.kind !== "del")
            .map((o, i) => chip(pb[(o as { b: number }).b], o.kind as "same" | "ins", i))}
        </div>
      </div>
    </div>
  );
}
