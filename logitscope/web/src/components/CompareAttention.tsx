// full-area two-column comparison of two input sentences — the whole instrument
// pointed at a pair of prompts instead of one generation.
//
// aimed at "sentence A vs sentence B (B has one extra whitespace)": both
// sentences are tokenized, diffed character by character, and each gets one
// prompt-only prefill whose capture feeds every readout the main panel has —
// next-token candidates, the logit lens up the stack, residual norms, and the
// attention square — plus the pairwise deltas the main panel has no use for.
//
// nothing here generates: the compare view is about the inputs themselves.

import { useEffect, useRef, useState } from "react";
import { fetchCompareAttn, runCompare } from "../actions";
import { attnRGB, deltaRGB, deltaShade, fitCanvas, probShade, type Ctx2D } from "../canvas";
import {
  attentionPeak,
  attentionStats,
  buildVerdict,
  compareAttentionChannels,
  compareDist,
  compareDistributions,
  diffMarks,
  editSummary,
  headAt,
  keyMass,
  probIn,
  residDelta,
  selfAttention,
  unionCandidates,
  type AttentionStats,
  type CompareDist,
  type DistMetrics,
  type Verdict,
} from "../compare";
import { entropyNats } from "../math";
import { currentModel, store, useStore, type CompareSide, type AttnMatrix } from "../store";
import { requestTexts, visible, visibleForDiff } from "../tokens";

const NEXT_K = 10; // candidate rows in the next-token table
const LENS_ROWS = 8; // layers shown per lens column
const PAD = 6; // attention canvas padding, mirrored by cellFrom()

type Side = "a" | "b";
type Matrix = { heads: number; n: number; data: Float32Array };
type Delta = { n: number; diff: Float32Array; maxAbs: number };

/** per-cell attention change. the delta heatmap's scale is the biggest move. */
function matrixDelta(a: Matrix, b: Matrix, head: number): Delta | null {
  if (a.n !== b.n || a.heads !== b.heads) return null;
  const n = a.n;
  const diff = new Float32Array(n * n);
  let maxAbs = 0;
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      const d = headAt(a.data, a.heads, n, head, i, j) - headAt(b.data, b.heads, n, head, i, j);
      diff[i * n + j] = d;
      if (Math.abs(d) > maxAbs) maxAbs = Math.abs(d);
    }
  return { n, diff, maxAbs };
}

export default function CompareAttention() {
  const sideA = useStore((s) => s.compare.a);
  const sideB = useStore((s) => s.compare.b);
  const running = useStore((s) => s.compare.running);
  const runId = useStore((s) => s.compare.runId);
  const layer = useStore((s) => s.compare.attnLayer);
  const head = useStore((s) => s.compare.attnHead);
  const attnOn = useStore((s) => s.compare.attn);
  const model = useStore((s) => currentModel(s));
  const texts = useStore((s) => s.tokenTexts);

  const [promptA, setPromptA] = useState("");
  const [promptB, setPromptB] = useState("");
  const [swapped, setSwapped] = useState(false);
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);

  const nLayers = model?.n_layers ?? 0;
  const nHeads = model?.n_heads ?? 0;

  const compare = () => {
    if (promptA.trim() && promptB.trim() && !running) void runCompare(promptA, promptB);
  };

  // "a" and "b" are the user's sentences; the columns can be shown either way
  // round, so everything downstream works on left/right plus the tag that
  // belongs to each side
  const leftSide = (swapped ? sideB : sideA) ?? null;
  const rightSide = (swapped ? sideA : sideB) ?? null;
  const leftTag: Side = swapped ? "b" : "a";
  const rightTag: Side = swapped ? "a" : "b";
  const leftText = leftTag === "a" ? promptA : promptB;
  const rightText = rightTag === "a" ? promptA : promptB;
  const setText = (side: Side, v: string): void => (side === "a" ? setPromptA : setPromptB)(v);

  const distLeft = leftSide ? compareDist(leftSide) : null;
  const distRight = rightSide ? compareDist(rightSide) : null;
  const metrics = distLeft && distRight ? compareDistributions(distLeft, distRight) : null;
  const attnStats =
    leftSide?.attnMatrix && rightSide?.attnMatrix
      ? attentionStats(leftSide.attnMatrix, rightSide.attnMatrix, head)
      : null;
  const delta =
    leftSide?.attnMatrix && rightSide?.attnMatrix
      ? matrixDelta(leftSide.attnMatrix, rightSide.attnMatrix, head)
      : null;
  const resid =
    leftSide?.residNorms && rightSide?.residNorms
      ? residDelta(leftSide.residNorms, rightSide.residNorms)
      : null;

  // the whole point of the view: what the edit did, in words. built from the
  // same numbers the panels draw, so it cannot drift from them.
  const edit =
    leftSide && rightSide
      ? editSummary(
          leftText,
          rightText,
          leftSide.session.promptIds,
          rightSide.session.promptIds,
          texts,
        )
      : null;
  const verdict = (() => {
    if (!edit || !leftSide || !rightSide) return null;
    const massLeft = leftSide.attnMatrix ? keyMass(leftSide.attnMatrix, head, -1) : null;
    const massRight = rightSide.attnMatrix ? keyMass(rightSide.attnMatrix, head, -1) : null;
    const peakLeft = leftSide.attnMatrix ? attentionPeak(leftSide.attnMatrix, head) : 0;
    const peakRight = rightSide.attnMatrix ? attentionPeak(rightSide.attnMatrix, head) : 0;

    // the lens is captured top-1 per layer, so "where the two disagree, and
    // whether they come back together" is a walk up the stack
    const lensA = leftSide.lens;
    const lensB = rightSide.lens;
    const layers = lensA && lensB ? Math.min(lensA.layers, lensB.layers) : 0;
    let diverge: number | null = null;
    let firstKeyShift: number | null = null;
    let lastKeyShift: number | null = null;
    let agree = 0;
    for (let l = 0; l < layers && lensA && lensB; l++) {
      if (lensA.ids[l * lensA.k] === lensB.ids[l * lensB.k]) {
        agree++;
      } else {
        if (diverge === null) diverge = l;
        if (firstKeyShift === null) firstKeyShift = l;
        lastKeyShift = l;
      }
    }

    // how far a one-character edit travelled through the layer's attention
    const channels =
      leftSide.attnMatrix && rightSide.attnMatrix
        ? compareAttentionChannels(
            {
              a: leftSide.attnMatrix,
              b: rightSide.attnMatrix,
              align: edit.align,
              created: edit.insertedTokens,
            },
            head,
            edit,
          )
        : null;

    return buildVerdict({
      promptA: leftSide.session.promptIds,
      promptB: rightSide.session.promptIds,
      texts,
      tagA: leftTag,
      tagB: rightTag,
      lensLabel: "the lens",
      measured: `L${layer} / ${nLayers || "?"} · ${head < 0 ? "mean head" : `head ${head}`}`,
      edit,
      channels,
      nextTop:
        distLeft && distRight
          ? {
              idA: distLeft.ids[0],
              idB: distRight.ids[0],
              pA: probIn(distLeft, distLeft.ids[0]),
              pB: probIn(distRight, distRight.ids[0]),
              js: metrics?.js ?? 0,
              tv: metrics?.tv ?? 0,
              sameTop: distLeft.ids[0] === distRight.ids[0],
            }
          : null,
      attention:
        massLeft && massRight && leftSide.attnMatrix && rightSide.attnMatrix
          ? {
              head,
              massA: massLeft,
              massB: massRight,
              stats: attnStats,
              peakA: peakLeft,
              peakB: peakRight,
              selfA: selfAttention(leftSide.attnMatrix, head),
              selfB: selfAttention(rightSide.attnMatrix, head),
            }
          : null,
      resid,
      residScale: leftSide.residNorms
        ? leftSide.residNorms.reduce((m, v) => Math.max(m, v), 0)
        : 0,
      lens:
        layers > 0 ? { diverge, agree, layers, firstKeyShift, lastKeyShift } : null,
    });
  })();

  // attention squares are per layer; a layer change invalidates both of them
  // and nothing else. ask for whatever is missing.
  useEffect(() => {
    if (attnOn) fetchCompareAttn();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [attnOn, layer, runId]);

  const haveAny = leftSide !== null || rightSide !== null;

  return (
    <div className="compare-full">
      <div className="compare-bar">
        <button onClick={() => store.set({ compareMode: false })}>← back</button>
        <button
          className="cmp-run"
          onClick={compare}
          disabled={running || !promptA.trim() || !promptB.trim()}
        >
          {running ? "running…" : "compare"}
        </button>
        <button
          className="cmp-preset"
          title="append one space to both sentences — the classic whitespace-perturbation pair"
          onClick={() => {
            setPromptA((p) => p + " ");
            setPromptB((p) => p + " ");
          }}
        >
          space␣both
        </button>
        <button
          className="cmp-preset"
          title="append a zero-width space (U+200B) to both — invisible in text, real to the tokenizer"
          onClick={() => {
            setPromptA((p) => p + "\u200b");
            setPromptB((p) => p + "\u200b");
          }}
        >
          zero-width
        </button>
        <button title="swap which sentence is drawn on the left" onClick={() => setSwapped((s) => !s)}>
          swap
        </button>
        <span className="hud-spacer" />
        <label>
          layer{" "}
          <select
            value={layer}
            disabled={!attnOn}
            onChange={(e) => store.setCompareAttn(Number(e.target.value), head)}
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
          <select
            value={head}
            disabled={!attnOn}
            onChange={(e) => store.setCompareAttn(layer, Number(e.target.value))}
          >
            <option value={-1}>mean</option>
            {Array.from({ length: nHeads }, (_, h) => (
              <option key={h} value={h}>
                {h}
              </option>
            ))}
          </select>
        </label>
        <label className="toggle" title="prompt-only attention squares (ctx² per head — the costly capture)">
          <input
            type="checkbox"
            checked={attnOn}
            onChange={(e) => store.setCompareAttnEnabled(e.target.checked)}
          />
          attn
        </label>
      </div>

      <div className="cmp-inputs">
        <label className="cmp-input">
          <span className={`cmp-tag cmp-tag-${leftTag}`}>{leftTag}</span>
          <textarea
            rows={1}
            value={leftText}
            placeholder="sentence a…"
            onChange={(e) => setText(leftTag, e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                compare();
              }
            }}
          />
        </label>
        <label className="cmp-input">
          <span className={`cmp-tag cmp-tag-${rightTag}`}>{rightTag}</span>
          <textarea
            rows={1}
            value={rightText}
            placeholder="sentence b…"
            onChange={(e) => setText(rightTag, e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                compare();
              }
            }}
          />
        </label>
      </div>

      {!haveAny ? (
        <div className="placeholder">
          enter two sentences and hit compare — the interesting pair differs by one whitespace
        </div>
      ) : (
        <div className="cmp-scroll">
          <DiffSection leftTag={leftTag} rightTag={rightTag} a={leftText} b={rightText} />
          <SummarySection
            left={leftSide}
            right={rightSide}
            leftTag={leftTag}
            rightTag={rightTag}
            distLeft={distLeft}
            distRight={distRight}
            metrics={metrics}
            stats={attnStats}
            resid={resid}
          />
          <NextTokenSection
            leftTag={leftTag}
            rightTag={rightTag}
            distLeft={distLeft}
            distRight={distRight}
            metrics={metrics}
            texts={texts}
          />
          <LensSection
            left={leftSide}
            right={rightSide}
            leftTag={leftTag}
            rightTag={rightTag}
            texts={texts}
          />
          <ResidSection left={leftSide} right={rightSide} leftTag={leftTag} rightTag={rightTag} resid={resid} />
          {attnOn && (
            <AttentionSection
              left={leftSide}
              right={rightSide}
              leftTag={leftTag}
              rightTag={rightTag}
              leftText={leftText}
              rightText={rightText}
              head={head}
              hover={hover}
              setHover={setHover}
              delta={delta}
              stats={attnStats}
              verdict={verdict}
            />
          )}
        </div>
      )}
    </div>
  );
}

// -- prompts: one chip per character, whitespace visible, diff highlighted --

function DiffSection({
  leftTag,
  rightTag,
  a,
  b,
}: {
  leftTag: Side;
  rightTag: Side;
  a: string;
  b: string;
}) {
  const da = visibleForDiff(a);
  const db = visibleForDiff(b);
  const marks = diffMarks(da, db);
  const changed = marks.changedA.filter(Boolean).length + marks.changedB.filter(Boolean).length;
  const render = (tag: Side, text: string): React.ReactNode => (
    <div className="cmp-diff-row" key={tag}>
      <span className={`cmp-tag cmp-tag-${tag}`}>{tag}</span>
      <div className="cmp-diff-chars">
        {charRuns(text, tag === "a" ? marks.changedA : marks.changedB).map((run) => (
          <span
            key={run.key}
            className={`chip ${run.changed ? `chip-${tag === "a" ? "del" : "ins"}` : "chip-ctx"}`}
          >
            {run.text}
          </span>
        ))}
      </div>
    </div>
  );
  // the label under each row is the tag's own sentence, so the rows read the
  // same way round as the columns below
  return (
    <section className="cmp-section">
      <div className="cmp-section-title">
        prompts — one chip per character, whitespace visible
        <span className="cmp-section-note">
          {changed === 0 ? "identical" : `${changed} chars differ`}
        </span>
      </div>
      {leftTag === "a" ? render("a", da) : render("b", db)}
      {rightTag === "b" ? render("b", db) : render("a", da)}
      <div className="cmp-diff-legend">
        <span className="chip chip-del">only in a</span>
        <span className="chip chip-ins">only in b</span>
        <span className="chip chip-ctx">aligned</span>
      </div>
    </section>
  );
}

/** group a string into [unchanged | changed] runs so each run is one chip. */
function charRuns(
  text: string,
  changed: boolean[],
): { key: number; text: string; changed: boolean }[] {
  const chars = [...text];
  const out: { key: number; text: string; changed: boolean }[] = [];
  let k = 0;
  while (k < chars.length) {
    const c = changed[k] === true;
    let end = k;
    while (end < chars.length && (changed[end] === true) === c) end++;
    out.push({ key: k, text: chars.slice(k, end).join(""), changed: c });
    k = end;
  }
  return out;
}

// -- the numbers, side by side --

function SummarySection({
  left,
  right,
  leftTag,
  rightTag,
  distLeft,
  distRight,
  metrics,
  stats,
  resid,
}: {
  left: CompareSide | null;
  right: CompareSide | null;
  leftTag: Side;
  rightTag: Side;
  distLeft: CompareDist | null;
  distRight: CompareDist | null;
  metrics: DistMetrics | null;
  stats: AttentionStats | null;
  resid: ReturnType<typeof residDelta>;
}) {
  const sameLen =
    left && right ? left.session.promptIds.length === right.session.promptIds.length : false;
  const cross = (() => {
    if (!distLeft || !distRight) return null;
    const p = distRight.probs[0];
    const q = distLeft.probs[0];
    return p > 0 && q > 0 ? p / q : null;
  })();
  const notes = [
    metrics ? (metrics.sameTop ? "same top-1 next token" : "different top-1 next token") : "",
    cross !== null ? `top-1 ratio ${cross.toFixed(2)}×` : "",
    left && right ? (sameLen ? "same token count" : "token count differs") : "",
  ].filter(Boolean);

  const rows: Array<[string, string, string]> = [
    ["tokens", left ? `${left.session.promptIds.length}` : "–", right ? `${right.session.promptIds.length}` : "–"],
    ["h of p(next) nats", distLeft ? distLeft.entropy.toFixed(3) : "–", distRight ? distRight.entropy.toFixed(3) : "–"],
    ["p(next) moved", metrics ? `${metrics.js.toFixed(4)} js` : "–", metrics ? `${metrics.tv.toFixed(4)} tv` : "–"],
    [
      "max ± next-token",
      metrics?.maxDelta ? `${fmtSigned(metrics.maxDelta.delta)} #${metrics.maxDelta.id}` : "–",
      metrics?.kl !== null && metrics !== null ? `${metrics.kl.toFixed(4)} kl` : "–",
    ],
    [
      "attention peak",
      stats ? stats.maxA.toFixed(3) : "–",
      stats ? stats.maxB.toFixed(3) : "–",
    ],
    [
      "attention |Δ|",
      stats?.at ? `${stats.maxAbs.toFixed(4)} @ (${stats.at.i},${stats.at.j})` : "–",
      stats ? `${stats.meanAbs.toFixed(5)} mean` : "–",
    ],
    [
      "residual ‖h‖",
      resid ? `${resid.maxAbs.toFixed(1)} @ ${resid.at === 0 ? "emb" : `L${resid.at - 1}`}` : "–",
      resid ? `${resid.meanAbs.toFixed(1)} mean |Δ|` : "–",
    ],
  ];

  return (
    <section className="cmp-section">
      <div className="cmp-section-title">
        summary
        {notes.length > 0 && <span className="cmp-section-note">{notes.join(" · ")}</span>}
      </div>
      <div className="cmp-table">
        <div className="cmp-tr cmp-tr-head">
          <span className="cmp-td-label" />
          <span className="cmp-td">
            <span className={`cmp-tag cmp-tag-${leftTag}`}>{leftTag}</span>
          </span>
          <span className="cmp-td">
            <span className={`cmp-tag cmp-tag-${rightTag}`}>{rightTag}</span>
          </span>
        </div>
        {rows.map(([label, a, b]) => (
          <div className="cmp-tr" key={label}>
            <span className="cmp-td-label">{label}</span>
            <span className="cmp-td">{a}</span>
            <span className="cmp-td">{b}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function fmtSigned(v: number): string {
  const s = v >= 0 ? "+" : "−";
  const m = Math.abs(v);
  return `${s}${m >= 0.001 ? m.toFixed(3) : m.toExponential(1)}`;
}

// -- next token after the prompt --

function NextTokenSection({
  leftTag,
  rightTag,
  distLeft,
  distRight,
  metrics,
  texts,
}: {
  leftTag: Side;
  rightTag: Side;
  distLeft: CompareDist | null;
  distRight: CompareDist | null;
  metrics: DistMetrics | null;
  texts: Record<number, string>;
}) {
  const single = distLeft && !distRight ? distLeft : distRight && !distLeft ? distRight : null;
  const ids =
    distLeft && distRight
      ? unionCandidates(distLeft, distRight, NEXT_K)
      : (single?.ids.slice(0, NEXT_K) ?? []);
  const idKey = ids.join(",");

  useEffect(() => {
    if (ids.length > 0) requestTexts(ids);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idKey]);

  if (ids.length === 0) {
    return (
      <section className="cmp-section">
        <div className="cmp-section-title">next token after the prompt</div>
        <div className="cmp-empty">no candidates yet — capture still in flight, or it failed</div>
      </section>
    );
  }

  const maxP = Math.max(...ids.map((id) => Math.max(probIn(distLeft, id), probIn(distRight, id))), 1e-9);
  const topLeft = distLeft?.ids[0];
  const topRight = distRight?.ids[0];

  return (
    <section className="cmp-section">
      <div className="cmp-section-title">
        next token after the prompt
        <span className="cmp-section-note">
          top-{ids.length} of the union · exact softmax over the prompt's last position
          {metrics ? ` · js ${metrics.js.toFixed(4)} nats` : ""}
        </span>
      </div>
      <div className="cmp-next">
        <div className="cmp-next-row cmp-next-head">
          <span className="cmp-next-tok" />
          <span className="cmp-bar-cell">
            <span className={`cmp-tag cmp-tag-${leftTag}`}>{leftTag}</span>
          </span>
          <span className="cmp-bar-cell">
            <span className={`cmp-tag cmp-tag-${rightTag}`}>{rightTag}</span>
          </span>
          <span className="cmp-next-delta">Δ</span>
        </div>
        {ids.map((id) => {
          const pl = probIn(distLeft, id);
          const pr = probIn(distRight, id);
          const d = pl - pr;
          const lensHit = distLeft?.lensTop?.id === id || distRight?.lensTop?.id === id;
          return (
            <div className="cmp-next-row" key={id}>
              <span
                className={lensHit ? "cmp-next-tok cmp-next-lens" : "cmp-next-tok"}
                title={`token id ${id}`}
              >
                {texts[id] !== undefined ? visible(texts[id]) : `#${id}`}
              </span>
              <span className={id === topLeft ? "cmp-bar-cell cmp-bar-top" : "cmp-bar-cell"}>
                <span className="cmp-bar cmp-bar-a" style={{ width: `${pctWidth(pl, maxP)}%` }} />
                <span className="cmp-bar-p">{pct(pl)}</span>
              </span>
              <span className={id === topRight ? "cmp-bar-cell cmp-bar-top" : "cmp-bar-cell"}>
                <span className="cmp-bar cmp-bar-b" style={{ width: `${pctWidth(pr, maxP)}%` }} />
                <span className="cmp-bar-p">{pct(pr)}</span>
              </span>
              <span className={Math.abs(d) < 0.005 ? "cmp-next-delta dim" : "cmp-next-delta"}>
                {fmtSigned(d)}
              </span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function pctWidth(p: number, max: number): number {
  return Math.max(1, (p / max) * 100);
}

function pct(p: number): string {
  // 0 here means "not in this side's top-k", not "impossible" — say so rather
  // than printing a bare 0e+0%
  if (p < 1e-5) return "<0.001%";
  if (p < 0.001) return `${(p * 100).toExponential(0)}%`;
  return `${(p * 100).toFixed(1)}%`;
}

// -- logit lens: where the next token forms, layer by layer --

function LensSection({
  left,
  right,
  leftTag,
  rightTag,
  texts,
}: {
  left: CompareSide | null;
  right: CompareSide | null;
  leftTag: Side;
  rightTag: Side;
  texts: Record<number, string>;
}) {
  const la = left?.lens ?? null;
  const lb = right?.lens ?? null;
  const lensKey = la && lb ? `${la.layers}:${lb.layers}:${la.ids.join(",")}:${lb.ids.join(",")}` : "";

  useEffect(() => {
    if (!la || !lb) return;
    const want: number[] = [];
    for (const s of [la, lb]) for (let l = 0; l < s.layers; l++) want.push(s.ids[l * s.k]);
    if (want.length > 0) requestTexts(want);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lensKey]);

  if (!la || !lb) {
    return (
      <section className="cmp-section">
        <div className="cmp-section-title">logit lens — where the next token forms</div>
        <div className="cmp-empty">
          {la || lb ? "waiting for the other side" : "no lens capture yet"}
        </div>
      </section>
    );
  }

  const common = Math.min(la.layers, lb.layers);
  let diverge = -1;
  let agree = 0;
  for (let l = 0; l < common; l++) {
    const same = la.ids[l * la.k] === lb.ids[l * lb.k];
    if (same) agree++;
    else if (diverge < 0) diverge = l;
  }
  const finalA = la.ids[(common - 1) * la.k];
  const finalB = lb.ids[(common - 1) * lb.k];

  return (
    <section className="cmp-section">
      <div className="cmp-section-title">
        logit lens — where the next token forms
        <span className="cmp-section-note">
          {la.layers === lb.layers ? `${common} layers` : `layer counts differ (${la.layers} vs ${lb.layers})`}
          {diverge < 0
            ? " · argmax agrees at every layer"
            : ` · first disagrees at ${diverge === 0 ? "emb" : `L${diverge - 1}`}`}
          {` · agree on ${agree}/${common} layer argmaxes`}
        </span>
      </div>
      <div className="cmp-lens">
        <LensTrace
          head={`${leftTag} · final → ${tokText(finalA, texts)}`}
          ids={la.ids}
          probs={la.probs}
          k={la.k}
          layers={la.layers}
          texts={texts}
        />
        <LensTrace
          head={`${rightTag} · final → ${tokText(finalB, texts)}`}
          ids={lb.ids}
          probs={lb.probs}
          k={lb.k}
          layers={lb.layers}
          texts={texts}
        />
        <div className="cmp-lens-trace">
          <div className="cmp-lens-head">layer · top guess · same?</div>
          {Array.from({ length: Math.min(LENS_ROWS, common) }, (_, r) => {
            const l = common - 1 - r;
            const ia = la.ids[l * la.k];
            const ib = lb.ids[l * lb.k];
            const same = ia === ib;
            return (
              <div className="cmp-lens-row" key={l}>
                <span className="cmp-lens-l">{l === 0 ? "emb" : `L${l - 1}`}</span>
                <span className={same ? "cmp-lens-tok same" : "cmp-lens-tok"}>
                  {tokText(ia, texts)}
                </span>
                <span className={same ? "cmp-lens-tok same" : "cmp-lens-tok"}>
                  {tokText(ib, texts)}
                </span>
                <span className={same ? "cmp-same" : "cmp-moved"}>{same ? "=" : "≠"}</span>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

function LensTrace({
  head,
  ids,
  probs,
  k,
  layers,
  texts,
}: {
  head: string;
  ids: Uint32Array;
  probs: Float32Array;
  k: number;
  layers: number;
  texts: Record<number, string>;
}) {
  const cells: React.ReactNode[] = [];
  for (let r = 0; r < Math.min(LENS_ROWS, layers); r++) {
    const l = layers - 1 - r;
    const id = ids[l * k];
    const p = probs[l * k];
    cells.push(
      <div className="cmp-lens-chip-row" key={l}>
        <span className="cmp-lens-l">{l === 0 ? "emb" : `L${l - 1}`}</span>
        <span
          className="cmp-lens-chip"
          style={{ background: probShade(p) }}
          title={`top-1 p ${(p * 100).toFixed(2)}%`}
        >
          {tokText(id, texts)}
        </span>
      </div>,
    );
  }
  return (
    <div className="cmp-lens-trace">
      <div className="cmp-lens-head">{head}</div>
      {cells}
    </div>
  );
}

function tokText(id: number, texts: Record<number, string>): string {
  const t = texts[id];
  return t !== undefined ? visible(t) : `#${id}`;
}

// -- residual stream norms --

function ResidSection({
  left,
  right,
  leftTag,
  rightTag,
  resid,
}: {
  left: CompareSide | null;
  right: CompareSide | null;
  leftTag: Side;
  rightTag: Side;
  resid: ReturnType<typeof residDelta>;
}) {
  const na = left?.residNorms ?? null;
  const nb = right?.residNorms ?? null;
  if (!na || !nb) {
    return (
      <section className="cmp-section">
        <div className="cmp-section-title">residual stream — ‖h‖ per layer</div>
        <div className="cmp-empty">
          {na || nb ? "waiting for the other side" : "no residual capture yet"}
        </div>
      </section>
    );
  }
  const layers = Math.min(na.length, nb.length);
  let max = 1e-9;
  for (let l = 0; l < layers; l++) max = Math.max(max, na[l], nb[l]);
  const scale = resid?.maxAbs || 1;

  return (
    <section className="cmp-section">
      <div className="cmp-section-title">
        residual stream — ‖h‖ per layer
        <span className="cmp-section-note">
          {resid
            ? `max |Δ| ${resid.maxAbs.toFixed(1)} at ${resid.at === 0 ? "emb" : `L${resid.at - 1}`} · mean ${resid.meanAbs.toFixed(1)} · norm scale ${max.toFixed(0)}`
            : ""}
        </span>
      </div>
      <div className="cmp-resid">
        {Array.from({ length: layers }, (_, l) => {
          const d = na[l] - nb[l];
          return (
            <div
              className="cmp-resid-col"
              key={l}
              title={`${l === 0 ? "emb" : `L${l - 1}`} · ${leftTag} ${na[l].toFixed(1)} · ${rightTag} ${nb[l].toFixed(1)} · Δ ${d.toFixed(1)}`}
            >
              <span className="cmp-resid-bars">
                <span className="cmp-resid-bar cmp-bar-a" style={{ height: `${(na[l] / max) * 100}%` }} />
                <span className="cmp-resid-bar cmp-bar-b" style={{ height: `${(nb[l] / max) * 100}%` }} />
              </span>
              <span
                className="cmp-resid-d"
                style={{ background: Math.abs(d) < 1e-6 ? "transparent" : deltaShade(d / scale, 0.8) }}
              />
              <span className="cmp-resid-l">{l === 0 ? "e" : l - 1}</span>
            </div>
          );
        })}
        <div className="cmp-resid-key">
          <span className="cmp-bar-swatch swatch-a" /> {leftTag}
          <span className="cmp-bar-swatch swatch-b" /> {rightTag}
          <span className="dim">cell tint = signed Δ (green {rightTag}↑, red {leftTag}↑)</span>
        </div>
      </div>
    </section>
  );
}

// -- attention: two squares, their delta, and where the mass sits --

function AttentionSection({
  left,
  right,
  leftTag,
  rightTag,
  head,
  hover,
  setHover,
  delta,
  stats,
  verdict,
}: {
  left: CompareSide | null;
  right: CompareSide | null;
  leftTag: Side;
  rightTag: Side;
  leftText: string;
  rightText: string;
  head: number;
  hover: { i: number; j: number } | null;
  setHover: (h: { i: number; j: number } | null) => void;
  delta: Delta | null;
  stats: AttentionStats | null;
  verdict: Verdict | null;
}) {
  const n = left?.attnMatrix?.n ?? right?.attnMatrix?.n ?? 0;
  const posText = (p: number): string => {
    const id = left?.session.promptIds[p];
    return id === undefined ? `#${p}` : `#${id}`;
  };
  const weight = (side: CompareSide | null): string => {
    if (!side?.attnMatrix || !hover) return "–";
    const m = side.attnMatrix;
    if (hover.i >= m.n || hover.j >= m.n) return "–";
    return headAt(m.data, m.heads, m.n, head, hover.i, hover.j).toFixed(4);
  };
  const status = (side: CompareSide | null): string =>
    side?.attnMatrix
      ? `${side.attnMatrix.n}×${side.attnMatrix.n}`
      : side?.capturePending
        ? "computing…"
        : "no data";

  return (
    <section className="cmp-section">
      <div className="cmp-section-title">
        attention over the prompt
        <span className="cmp-section-note">
          {status(left)} vs {status(right)}
          {stats?.at
            ? ` · max |Δ| ${stats.maxAbs.toFixed(4)} at (${stats.at.i},${stats.at.j}) · mean ${stats.meanAbs.toFixed(5)}`
            : ""}
        </span>
      </div>
      <div className="cmp-attn">
        <div className="cmp-attn-grid">
          <MatrixPane
            side={left}
            tag={leftTag}
            head={head}
            hover={hover}
            setHover={setHover}
            own="left"
          />
          <MatrixPane
            side={right}
            tag={rightTag}
            head={head}
            hover={hover}
            setHover={setHover}
            own="right"
          />
          <div className="cmp-attn-pane">
            <div className="cmp-pane-head">
              Δ attention ({leftTag}−{rightTag})
              {delta ? <span className="dim"> · scale {delta.maxAbs.toFixed(3)}</span> : null}
            </div>
            {delta ? (
              <DeltaCanvas delta={delta} row={hover?.i ?? null} />
            ) : (
              <div className="cmp-empty">needs equal context length on both sides</div>
            )}
          </div>
          {hover && n > 0 && (
            <div className="cmp-attn-readout">
              row {hover.i} · q {posText(hover.i)} ← k {posText(hover.j)} · {leftTag}{" "}
              {weight(left)} · {rightTag} {weight(right)}
            </div>
          )}
          <div className="cmp-mass">
            <MassRow side={left} tag={leftTag} head={head} pos={posText} />
            <MassRow side={right} tag={rightTag} head={head} pos={posText} />
          </div>
        </div>
        <VerdictPanel verdict={verdict} />
      </div>
    </section>
  );
}

/** the verdict column. it takes the empty space to the right of the two squares
 * and answers, in words, what turning a into b did. */
function VerdictPanel({ verdict }: { verdict: Verdict | null }) {
  const [copied, setCopied] = useState(false);

  if (!verdict) {
    return (
      <aside className="cmp-verdict">
        <div className="cmp-verdict-head">
          <span className="cmp-verdict-title">结论</span>
        </div>
        <div className="cmp-empty">等两侧 capture 到齐，就能给出结论</div>
      </aside>
    );
  }

  const sevClass =
    verdict.severity.level === "major"
      ? "cmp-sev-major"
      : verdict.severity.level === "moderate"
        ? "cmp-sev-mid"
        : "cmp-sev-low";

  const copy = (): void => {
    // verdict.text already carries the two prompts' answer, the scope and the
    // evidence in reading order
    void navigator.clipboard?.writeText(verdict.text).then(
      () => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1400);
      },
      () => setCopied(false),
    );
  };

  return (
    <aside className="cmp-verdict">
      <div className="cmp-verdict-head">
        <span className="cmp-verdict-title">结论</span>
        <span className={`cmp-sev ${sevClass}`}>{verdict.severity.label}</span>
        {verdict.scope.kind !== "none" && (
          <span className={`cmp-scope cmp-scope-${verdict.scope.kind}`}>
            {verdict.scope.kind === "global"
              ? "global"
              : verdict.scope.kind === "mixed"
                ? "mixed"
                : "local"}
          </span>
        )}
        <span className="cmp-measured" title="结论只对这个切片成立">
          {verdict.measured}
        </span>
        <span className="hud-spacer" />
        <button className="cmp-verdict-copy" onClick={copy} title="把这段结论连同两个输入复制走">
          {copied ? "已复制" : "copy"}
        </button>
      </div>

      {/* the question this whole view exists to ask, answered before anything
          else: can one character move the model's attention globally? */}
      <div className="cmp-verdict-answer">
        <p className="cmp-answer-line">{verdict.answer}</p>
        <p className="cmp-conclusion-line">{verdict.conclusion}</p>
      </div>

      <div className="cmp-verdict-analysis">
        <div className="cmp-verdict-change">edit: {verdict.change}</div>
        <ul className="cmp-obs">
          {verdict.observations.map((o, i) => (
            <li key={i} className={`cmp-obs-${o.level}`}>
              <span className="cmp-obs-dot" />
              {o.text}
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}

function MatrixPane({
  side,
  tag,
  head,
  hover,
  setHover,
  own,
}: {
  side: CompareSide | null;
  tag: Side;
  head: number;
  hover: { i: number; j: number } | null;
  setHover: (h: { i: number; j: number } | null) => void;
  own: "left" | "right";
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const matrix = side?.attnMatrix ?? null;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const draw = (): void => {
      const c = fitCanvas(canvas);
      if (c) drawMatrix(c, matrix, head, hover, own);
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [matrix, head, hover, own]);
  return (
    <div className="cmp-attn-pane">
      <div className="cmp-pane-head">
        <span className={`cmp-tag cmp-tag-${tag}`}>{tag}</span>{" "}
        {matrix ? `${matrix.n}×${matrix.n}` : side?.capturePending ? "computing…" : "no data"}
        {side?.error ? <span className="cmp-err"> {side.error}</span> : null}
      </div>
      <canvas
        ref={ref}
        className="cmp-matrix"
        onMouseMove={(ev) => {
          if (!matrix) return;
          setHover(cellFrom(ev, matrix.n));
        }}
        onMouseLeave={() => setHover(null)}
      />
    </div>
  );
}

function DeltaCanvas({ delta, row }: { delta: Delta; row: number | null }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const draw = (): void => {
      const c = fitCanvas(canvas);
      if (c) drawDelta(c, delta, row);
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [delta, row]);
  return <canvas ref={ref} className="cmp-matrix" />;
}

function MassRow({
  side,
  tag,
  head,
  pos,
}: {
  side: CompareSide | null;
  tag: Side;
  head: number;
  pos: (p: number) => string;
}) {
  const mass = side?.attnMatrix ? keyMass(side.attnMatrix, head, -1) : null;
  if (!mass) {
    return (
      <div className="cmp-mass-row">
        <span className={`cmp-tag cmp-tag-${tag}`}>{tag}</span>
        <span className="dim">no attention data</span>
      </div>
    );
  }
  const total = mass.reduce((s, v) => s + v, 0);
  const norm = Array.from(mass, (v) => (total > 0 ? v / total : 0));
  const order = norm.map((_, i) => i).sort((x, y) => norm[y] - norm[x]);  const top = order.slice(0, 6);
  const max = norm[top[0]] || 1e-9;

  return (
    <div className="cmp-mass-row">
      <span className={`cmp-tag cmp-tag-${tag}`}>{tag}</span>
      <span className="cmp-mass-head">
        mean mass per key · H {entropyNats(norm).toFixed(2)} nats
      </span>
      {top.map((p) => (
        <span className="cmp-mass-cell" key={p} title={`position ${p} (${pos(p)})`}>
          <span className="cmp-mass-p">{p}</span>
          <span className="cmp-mass-bar" style={{ width: `${Math.max(2, (norm[p] / max) * 44)}px` }} />
          <span className="cmp-mass-v">{norm[p].toFixed(3)}</span>
        </span>
      ))}
    </div>
  );
}

// -- attention canvases: square, lower-triangular, dpr-crisp --

function drawMatrix(
  c: Ctx2D,
  matrix: AttnMatrix | null,
  head: number,
  hover: { i: number; j: number } | null,
  own: "left" | "right",
): void {
  const { ctx, w, h } = c;
  ctx.clearRect(0, 0, w, h);
  if (!matrix || matrix.n === 0) return;
  const n = matrix.n;
  const size = Math.min(w, h) - 2 * PAD;
  if (size <= 0) return;

  const off = document.createElement("canvas");
  off.width = n;
  off.height = n;
  const octx = off.getContext("2d");
  if (!octx) return;
  const img = octx.createImageData(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      const [r, g, b] = attnRGB(headAt(matrix.data, matrix.heads, n, head, i, j));
      const o = (i * n + j) * 4;
      img.data[o] = r;
      img.data[o + 1] = g;
      img.data[o + 2] = b;
      img.data[o + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, PAD, PAD, size, size);

  if (!hover || hover.i >= n || hover.j > hover.i) return;
  const cell = size / n;
  if (own === "left") {
    // the crosshair belongs to whichever square the pointer is on
    ctx.strokeStyle = "rgba(201,228,212,0.8)";
    ctx.strokeRect(PAD + hover.j * cell, PAD + hover.i * cell, cell, cell);
    ctx.strokeStyle = "rgba(201,228,212,0.5)";
  } else {
    // the sibling echoes the query row and key column in amber
    ctx.strokeStyle = "rgba(255,200,87,0.5)";
  }
  ctx.strokeRect(PAD, PAD + hover.i * cell, size, cell);
  ctx.strokeRect(PAD + hover.j * cell, PAD, cell, size);
}

function drawDelta(c: Ctx2D, delta: Delta, row: number | null): void {
  const { ctx, w, h } = c;
  ctx.clearRect(0, 0, w, h);
  const n = delta.n;
  if (n === 0) return;
  const size = Math.min(w, h) - 2 * PAD;
  if (size <= 0) return;

  const off = document.createElement("canvas");
  off.width = n;
  off.height = n;
  const octx = off.getContext("2d");
  if (!octx) return;
  const img = octx.createImageData(n, n);
  const scale = delta.maxAbs || 1;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      // positive = the left column attends more here, so it reads red
      const [r, g, b] = deltaRGB(-delta.diff[i * n + j] / scale);
      const o = (i * n + j) * 4;
      img.data[o] = r;
      img.data[o + 1] = g;
      img.data[o + 2] = b;
      img.data[o + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, PAD, PAD, size, size);

  if (row !== null && row < n) {
    const cell = size / n;
    ctx.strokeStyle = "rgba(255,200,87,0.5)";
    ctx.strokeRect(PAD, PAD + row * cell, size, cell);
  }
}

/** canvas pixel -> matrix cell. mirrors drawMatrix's geometry exactly. */
function cellFrom(
  ev: { clientX: number; clientY: number; currentTarget: HTMLCanvasElement },
  n: number,
): { i: number; j: number } | null {
  const rect = ev.currentTarget.getBoundingClientRect();
  const size = Math.min(rect.width, rect.height) - 2 * PAD;
  if (size <= 0) return null;
  const j = Math.floor(((ev.clientX - rect.left - PAD) / size) * n);
  const i = Math.floor(((ev.clientY - rect.top - PAD) / size) * n);
  if (i < 0 || j < 0 || i >= n || j > i) return null;
  return { i, j };
}
