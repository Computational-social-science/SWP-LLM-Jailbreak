// hand-rolled store. one immutable state object, subscribe/notify, and
// useSyncExternalStore on the react side. zustand is nice but this is ~60
// lines and i know exactly what it does.

import { useSyncExternalStore } from "react";
import type { StepData } from "./client";
import type { LoadProgress, ModelInfo, NodeDict, SamplingParams, SessionDict } from "./types";
import { DEFAULT_PARAMS, ROOT } from "./types";

export interface AttnMatrix {
  forNode: number; // the node whose scrub produced it
  layer: number;
  heads: number;
  n: number; // context length covered: positions 0..n-1
  data: Float32Array; // (heads, n, n)
}

export interface SessionInfo {
  id: string;
  model: string;
  prompt: string;
  promptIds: number[];
}

/** one column of the compare view: an input sentence + everything one capture
 * pass over that sentence tells us. */
export interface CompareSide {
  session: SessionInfo;
  /** the whole prompt-only square, at CompareState.attnLayer */
  attnMatrix: AttnMatrix | null;
  attnPending: boolean;
  /** next-token candidates after the prompt, from the prefill's last position */
  topkIds: Uint32Array | null;
  topkLogits: Float32Array | null;
  /** logit lens down the stack at that same position: where the guess forms */
  lens: { layers: number; k: number; ids: Uint32Array; probs: Float32Array } | null;
  /** residual-stream norm per layer, embeddings first */
  residNorms: Float32Array | null;
  capturePending: boolean;
  error: string | null;
}

export interface CompareState {
  a: CompareSide | null;
  b: CompareSide | null;
  attnLayer: number;
  attnHead: number; // -1 = mean over heads
  /** attention squares are ctx² floats per head — off means metrics only */
  attn: boolean;
  running: boolean;
  runId: number; // bumps per run so stale replies can be dropped
}

export interface State {
  models: ModelInfo[];
  modelName: string;
  modelLoading: boolean;
  loadProgress: LoadProgress | null; // for the model currently being loaded
  conn: "connecting" | "open" | "closed";
  session: SessionInfo | null;
  nodes: Record<number, NodeDict>;
  steps: Record<number, StepData>;
  tip: number;
  path: number[]; // node ids from just under the root to the tip
  selected: number | null;
  hovered: number | null;
  hoverTokenId: number | null; // waterfall cell hover -> chips glow
  hoverPos: number | null; // context position under the mouse -> ribbons
  generating: boolean;
  params: SamplingParams;
  capture: { lens: boolean; residNorms: boolean };
  bottomTab: "waterfall" | "attention" | "diff";
  diffOther: number | null; // which other tip the diff view compares against
  compare: CompareState;
  compareMode: boolean; // full-area two-column input comparison
  attnLayer: number;
  attnHead: number; // -1 = mean over heads
  attnMatrix: AttnMatrix | null;
  attnPending: boolean;
  tokenTexts: Record<number, string>; // token id -> decoded text, per model
  stats: { tokens: number; bytes: number; tokPerSec: number; bytesPerTok: number };
  error: string | null;
}

const initial: State = {
  models: [],
  modelName: "",
  modelLoading: false,
  loadProgress: null,
  conn: "closed",
  session: null,
  nodes: {},
  steps: {},
  tip: ROOT,
  path: [],
  selected: null,
  hovered: null,
  hoverTokenId: null,
  hoverPos: null,
  generating: false,
  params: DEFAULT_PARAMS,
  capture: { lens: true, residNorms: true },
  bottomTab: "waterfall",
  diffOther: null,
  compare: { a: null, b: null, attnLayer: 0, attnHead: -1, attn: true, running: false, runId: 0 },
  compareMode: false,
  attnLayer: 0,
  attnHead: -1,
  attnMatrix: null,
  attnPending: false,
  tokenTexts: {},
  stats: { tokens: 0, bytes: 0, tokPerSec: 0, bytesPerTok: 0 },
  error: null,
};

/** leaves of the session tree, oldest first — one per timeline. */
export function branchTips(nodes: Record<number, NodeDict>): number[] {
  return Object.values(nodes)
    .filter((n) => n.id !== ROOT && n.children.length === 0)
    .map((n) => n.id)
    .sort((a, b) => a - b);
}

export function pathTo(nodes: Record<number, NodeDict>, id: number): number[] {
  const out: number[] = [];
  let cur = id;
  while (cur !== ROOT) {
    const n = nodes[cur];
    if (!n || n.parent === null) break;
    out.push(cur);
    cur = n.parent;
  }
  return out.reverse();
}

export class Store {
  state: State = initial;
  private listeners = new Set<() => void>();
  private lastTokAt: number | null = null;
  private emaDt: number | null = null;

  subscribe = (fn: () => void): (() => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };

  set(patch: Partial<State>): void {
    this.state = { ...this.state, ...patch };
    for (const fn of this.listeners) fn();
  }

  // -- session --

  resetSession(sess: SessionDict): void {
    const nodes: Record<number, NodeDict> = {};
    for (const n of sess.nodes) nodes[n.id] = n;
    const modelChanged = this.state.session !== null && this.state.session.model !== sess.model;
    this.set({
      ...(modelChanged ? { tokenTexts: {} } : {}), // texts are per-tokenizer
      session: { id: sess.id, model: sess.model, prompt: sess.prompt, promptIds: sess.prompt_ids },
      nodes,
      steps: {},
      tip: ROOT,
      path: [],
      selected: null,
      hovered: null,
      hoverTokenId: null,
      diffOther: null,
      attnMatrix: null,
      stats: { tokens: 0, bytes: 0, tokPerSec: 0, bytesPerTok: 0 },
      error: null,
    });
    this.lastTokAt = null;
    this.emaDt = null;
  }

  // -- streaming --

  startGeneration(): void {
    this.lastTokAt = null;
    this.set({ generating: true, error: null });
  }

  applyToken(node: NodeDict, step: StepData, at: number = performance.now()): void {
    const nodes = { ...this.state.nodes, [node.id]: node };
    if (node.parent !== null && nodes[node.parent]) {
      const parent = nodes[node.parent];
      if (!parent.children.includes(node.id))
        nodes[node.parent] = { ...parent, children: [...parent.children, node.id] };
    }
    if (this.lastTokAt !== null) {
      const dt = at - this.lastTokAt;
      this.emaDt = this.emaDt === null ? dt : 0.8 * this.emaDt + 0.2 * dt;
    }
    this.lastTokAt = at;
    const s = this.state.stats;
    const tokens = s.tokens + 1;
    const bytes = s.bytes + step.bytes;
    this.set({
      nodes,
      steps: { ...this.state.steps, [node.id]: step },
      tip: node.id,
      path: pathTo(nodes, node.id),
      stats: {
        tokens,
        bytes,
        tokPerSec: this.emaDt ? 1000 / this.emaDt : 0,
        bytesPerTok: bytes / tokens,
      },
    });
  }

  finishGeneration(): void {
    this.set({ generating: false });
  }

  applyCapture(nodeId: number, step: StepData): void {
    const prev = this.state.steps[nodeId];
    // scrubs fill gaps but never clobber what the live stream already sent:
    // a cheap attention-only scrub shouldn't shrink the token's top-32
    const merged: StepData = { ...step, ...prev, attn: step.attn ?? prev?.attn, bytes: prev?.bytes ?? step.bytes };
    const patch: Partial<State> = { steps: { ...this.state.steps, [nodeId]: merged }, attnPending: false };
    const attn = step.attn?.[0];
    if (attn && attn.q === attn.kv) {
      // a scrub recompute runs one forward over the whole prefix, so the frame
      // is the full square matrix — everything the heatmap needs in one shot
      patch.attnMatrix = {
        forNode: nodeId,
        layer: attn.layer,
        heads: attn.heads,
        n: attn.q,
        data: attn.data,
      };
    }
    this.set(patch);
  }

  // -- compare (two prompts side by side) --

  /** which column a session belongs to, if any. compare sessions are cheap
   * throwaways, so a capture reply naming one always belongs to the view. */
  compareSideOf(sessionId: string): "a" | "b" | null {
    if (!sessionId) return null;
    const c = this.state.compare;
    if (c.a?.session.id === sessionId) return "a";
    if (c.b?.session.id === sessionId) return "b";
    return null;
  }

  initCompareSide(side: "a" | "b", sess: SessionDict): void {
    const s: CompareSide = {
      session: { id: sess.id, model: sess.model, prompt: sess.prompt, promptIds: sess.prompt_ids },
      attnMatrix: null,
      attnPending: false,
      topkIds: null,
      topkLogits: null,
      lens: null,
      residNorms: null,
      capturePending: false,
      error: null,
    };
    this.set({ compare: { ...this.state.compare, [side]: s } });
  }

  /** one capture reply lands here: a full-square attention frame becomes the
   * heatmap, the rest is what the two columns compare numerically. */
  applyCompareCapture(side: "a" | "b", nodeId: number, step: StepData): void {
    const c = this.state.compare;
    const s = c[side];
    if (!s) return;
    const attn = step.attn?.[0];
    let attnMatrix = s.attnMatrix;
    if (attn && attn.q === attn.kv) {
      attnMatrix = {
        forNode: nodeId,
        layer: attn.layer,
        heads: attn.heads,
        n: attn.q,
        data: attn.data,
      };
    }
    const next: CompareSide = {
      ...s,
      attnMatrix,
      attnPending: false,
      capturePending: false,
      // a prompt-only capture carries the full top-k, so these are never
      // clobbered by a partial one — only filled in
      topkIds: step.topkIds ?? s.topkIds,
      topkLogits: step.topkLogits ?? s.topkLogits,
      lens: step.lens ?? s.lens,
      residNorms: step.residNorms ?? s.residNorms,
      error: null,
    };
    this.set({ compare: { ...c, [side]: next } });
  }

  setComparePending(side: "a" | "b", pending: boolean): void {
    const c = this.state.compare;
    const s = c[side];
    if (!s) return;
    this.set({ compare: { ...c, [side]: { ...s, attnPending: pending, capturePending: pending } } });
  }

  setCompareError(side: "a" | "b", error: string | null): void {
    const c = this.state.compare;
    const s = c[side];
    if (!s) return;
    this.set({
      compare: { ...c, [side]: { ...s, error, attnPending: false, capturePending: false } },
    });
  }

  setCompareAttn(layer: number, head: number): void {
    const c = this.state.compare;
    const stale = layer !== c.attnLayer;
    this.set({
      compare: {
        ...c,
        attnLayer: layer,
        attnHead: head,
        // the square is per-layer; the top-k, lens and residual captures are
        // not, so only the heatmaps go stale on a layer change
        ...(stale
          ? {
              a: c.a ? { ...c.a, attnMatrix: null } : null,
              b: c.b ? { ...c.b, attnMatrix: null } : null,
            }
          : {}),
      },
    });
  }

  /** attention squares are the expensive capture; off leaves the numeric
   * comparison (next-token, lens, residual) as the whole view. */
  setCompareAttnEnabled(on: boolean): void {
    const c = this.state.compare;
    if (c.attn === on) return;
    this.set({
      compare: {
        ...c,
        attn: on,
        a: c.a ? { ...c.a, attnMatrix: null, attnPending: false } : null,
        b: c.b ? { ...c.b, attnMatrix: null, attnPending: false } : null,
      },
    });
  }

  // -- ui --

  /** time travel: make another branch's tip the current path. steps and nodes
   * stay put — nothing about the old timeline is forgotten, just deselected. */
  checkout(id: number): void {
    if (!(id in this.state.nodes)) return;
    this.set({ tip: id, path: pathTo(this.state.nodes, id), selected: null, hovered: null });
  }

  select(id: number | null): void {
    this.set({ selected: id });
  }

  hover(id: number | null): void {
    this.set({ hovered: id });
  }

  hoverToken(tokenId: number | null): void {
    this.set({ hoverTokenId: tokenId });
  }

  setParams(patch: Partial<SamplingParams>): void {
    this.set({ params: { ...this.state.params, ...patch } });
  }

  setCapture(patch: Partial<State["capture"]>): void {
    this.set({ capture: { ...this.state.capture, ...patch } });
  }

  setAttnView(layer: number, head: number): void {
    const stale = layer !== this.state.attnLayer;
    this.set({ attnLayer: layer, attnHead: head, ...(stale ? { attnMatrix: null } : {}) });
  }

  mergeTexts(texts: Record<number, string>): void {
    this.set({ tokenTexts: { ...this.state.tokenTexts, ...texts } });
  }

  setError(err: string | null): void {
    this.set({ error: err });
  }
}

export const store = new Store();

export function useStore<T>(sel: (s: State) => T): T {
  return useSyncExternalStore(store.subscribe, () => sel(store.state));
}

/** the node the detail views should show: hover wins, then click, then tip. */
export function focusNode(s: State): number | null {
  if (s.hovered !== null) return s.hovered;
  if (s.selected !== null) return s.selected;
  return s.tip !== ROOT ? s.tip : null;
}

export function currentModel(s: State): ModelInfo | undefined {
  const name = s.session?.model ?? s.modelName;
  return s.models.find((m) => m.name === name);
}
