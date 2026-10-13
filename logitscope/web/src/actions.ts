// the io layer: owns the socket, talks rest, mutates the store. components
// call these and render state — nothing else touches the network.

import * as api from "./api";
import { ScopeSocket } from "./client";
import { branchTips, pathTo, store } from "./store";
import { requestTexts } from "./tokens";
import type { CaptureConfig } from "./types";
import { ROOT } from "./types";

export const socket = new ScopeSocket({
  onStatus: (s) => store.set({ conn: s }),
  onToken: (_sessionId, node, step) => {
    store.applyToken(node, step);
    if (step.topkIds) requestTexts(step.topkIds);
    if (step.lens) requestTexts(step.lens.ids);
  },
  onDone: () => store.finishGeneration(),
  onCapture: (sessionId, nodeId, _cached, step) => {
    const side = store.compareSideOf(sessionId);
    if (side) store.applyCompareCapture(side, nodeId, step);
    else store.applyCapture(nodeId, step);
  },
  onError: (_sessionId, err) => {
    store.setError(err);
    store.set({ generating: false, attnPending: false });
    // a failed compare pass should say so in the column it belonged to
    for (const side of ["a", "b"] as const) {
      if (store.state.compare[side]?.capturePending) store.setCompareError(side, err);
    }
  },
});

export async function init(): Promise<void> {
  socket.connect();
  try {
    const models = await api.listModels();
    const current = store.state.modelName;
    const pick = models.find((m) => m.name === current) ?? models.find((m) => m.loaded);
    store.set({ models, modelName: pick?.name ?? models[0]?.name ?? "" });
  } catch (e) {
    store.setError(e instanceof Error ? e.message : String(e));
  }
}

export async function loadModel(name: string): Promise<void> {
  store.set({ modelLoading: true, error: null, loadProgress: null });
  // a qwen download is minutes, not seconds — poll the server so the button
  // can say how far along it is instead of just spinning
  const poll = setInterval(() => {
    void api
      .loadProgress()
      .then((all) => {
        if (all[name]) store.set({ loadProgress: all[name] });
      })
      .catch(() => {}); // progress is cosmetic
  }, 500);
  try {
    await api.loadModel(name);
    const models = await api.listModels();
    store.set({ models, modelName: name });
  } catch (e) {
    store.setError(e instanceof Error ? e.message : String(e));
  } finally {
    clearInterval(poll);
    store.set({ modelLoading: false, loadProgress: null });
  }
}

function captureCfg(): CaptureConfig {
  const c = store.state.capture;
  return { topk_logits: 32, lens: c.lens, resid_norms: c.residNorms };
}

export async function newSession(prompt: string): Promise<void> {
  const model = store.state.modelName;
  if (!model) {
    store.setError("no model selected");
    return;
  }
  try {
    const sess = await api.createSession(model, prompt);
    store.resetSession(sess);
    requestTexts(sess.prompt_ids); // attention view labels prompt positions too
    store.startGeneration();
    socket.generate(sess.id, ROOT, store.state.params, captureCfg());
  } catch (e) {
    store.setError(e instanceof Error ? e.message : String(e));
  }
}

/** keep generating from the current tip (or a selected ancestor). */
export function continueFrom(nodeId?: number): void {
  const s = store.state;
  if (!s.session || s.generating) return;
  store.startGeneration();
  socket.generate(s.session.id, nodeId ?? s.tip, s.params, captureCfg());
}

export async function saveSession(): Promise<void> {
  const s = store.state.session;
  if (!s) return;
  try {
    const text = await api.exportSession(s.id);
    const a = document.createElement("a");
    const url = URL.createObjectURL(new Blob([text], { type: "application/jsonl" }));
    a.href = url;
    a.download = `logitscope-${s.id}.jsonl`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    store.setError(e instanceof Error ? e.message : String(e));
  }
}

export async function openSession(file: File): Promise<void> {
  try {
    const sess = await api.importSession(await file.text());
    store.resetSession(sess);
    requestTexts(sess.prompt_ids);
    // land on the longest timeline so the stream isn't empty
    const nodes = store.state.nodes;
    const tips = branchTips(nodes);
    if (tips.length > 0) {
      const longest = tips.reduce((a, b) =>
        pathTo(nodes, b).length > pathTo(nodes, a).length ? b : a,
      );
      store.checkout(longest);
    }
  } catch (e) {
    store.setError(e instanceof Error ? e.message : String(e));
  }
}

/** re-request the generation-time capture for every node on the current path.
 * a loaded session arrives as bare nodes — this refills lens/topk for every
 * view. misses recompute one forward each and warm the lru, so replaying a
 * second time (or scrubbing after) is all cache hits. */
export function replaySession(): void {
  const s = store.state;
  if (!s.session || s.generating) return;
  for (const id of s.path) socket.capture(s.session.id, id, captureCfg());
}

/** fork: regrow from nodeId's prefix with tokenId forced in its place. the
 * new branch streams like any generation and becomes the current path. */
export function forkAt(nodeId: number, tokenId: number): void {
  const s = store.state;
  if (!s.session || s.generating) return;
  store.select(null); // focus should follow the new timeline, not the old one
  store.startGeneration();
  socket.fork(s.session.id, nodeId, tokenId, s.params, captureCfg());
}

/** ask for the attention square at the tip via the scrub path. one forward
 * over the whole prefix returns (heads, ctx, ctx) — the entire heatmap in a
 * single frame, and the lru makes the second ask free. */
export function fetchAttn(): void {
  const s = store.state;
  if (!s.session || s.generating || s.attnPending || s.tip === ROOT) return;
  store.set({ attnPending: true });
  socket.capture(s.session.id, s.tip, { topk_logits: 8, attn_layers: [s.attnLayer] });
}

/** compare two input sentences (e.g. one differs by a single whitespace) side
 * by side: tokenize both, then pull one prompt-only capture per side. no
 * generation — the compare view is about the inputs themselves. */
export async function runCompare(promptA: string, promptB: string): Promise<void> {
  const s = store.state;
  if (!s.modelName) {
    store.setError("no model selected");
    return;
  }
  if (s.compare.running) return;
  const run = s.compare.runId + 1;
  store.set({
    error: null,
    compare: {
      a: null,
      b: null,
      attnLayer: s.compare.attnLayer,
      attnHead: s.compare.attnHead,
      attn: s.compare.attn,
      running: true,
      runId: run,
    },
  });
  try {
    const [sessA, sessB] = await Promise.all([
      api.createSession(s.modelName, promptA),
      api.createSession(s.modelName, promptB),
    ]);
    store.initCompareSide("a", sessA);
    store.initCompareSide("b", sessB);
    requestTexts(sessA.prompt_ids);
    requestTexts(sessB.prompt_ids);
    // both sides are valuable on their own, so they are two independent
    // requests: one side failing leaves the other column intact
    captureCompareSide("a", run);
    captureCompareSide("b", run);
  } catch (e) {
    store.setError(e instanceof Error ? e.message : String(e));
  } finally {
    store.set({ compare: { ...store.state.compare, running: false } });
  }
}

function compareRunId(): number {
  return store.state.compare.runId;
}

/** the compare view compares whole input sentences, not generated text, so its
 * capture asks for everything one prefill over the prompt can tell us: the
 * prompt-only attention square, the next-token candidates, the logit lens down
 * the stack and the residual norms. one forward, one reply per side. */
function compareCaptureCfg(): CaptureConfig {
  const c = store.state.compare;
  // attn_heads stays unset: the head picker is client-side, so every head has
  // to be in the frame for the picker to mean anything
  return {
    topk_logits: 32,
    lens: true,
    lens_topk: 8,
    resid_norms: true,
    ...(c.attn ? { attn_layers: [c.attnLayer] } : {}),
  };
}

/** one side's capture. a run is identified by its runId so a reply from a
 * comparison the user already replaced is dropped instead of landing in the
 * new columns. */
function captureCompareSide(side: "a" | "b", run: number): void {
  if (run !== compareRunId()) return;
  const s = store.state.compare[side];
  if (!s || s.capturePending) return;
  store.setComparePending(side, true);
  socket.capturePrompt(s.session.id, compareCaptureCfg());
}

/** re-ask for the attention squares after a layer change (or after attention
 * was switched back on). the numeric capture is layer-independent, so this is
 * the only thing a layer change costs. */
export function fetchCompareAttn(): void {
  if (!store.state.compare.attn) return;
  const run = compareRunId();
  for (const side of ["a", "b"] as const) {
    const s = store.state.compare[side];
    if (!s || s.attnPending || s.attnMatrix) continue;
    captureCompareSide(side, run);
  }
}
