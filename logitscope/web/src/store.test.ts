import { describe, expect, it } from "vitest";
import type { StepData } from "./client";
import { Store, branchTips, focusNode, pathTo } from "./store";
import type { NodeDict, SessionDict } from "./types";

function node(id: number, parent: number | null, over: Partial<NodeDict> = {}): NodeDict {
  return {
    id,
    parent,
    token_id: id === 0 ? null : 100 + id,
    text: `t${id}`,
    logprob: -0.5,
    rank: 1,
    entropy: 1.0,
    children: [],
    ...over,
  };
}

const sess: SessionDict = {
  id: "abc",
  model: "tiny-random",
  prompt: "hi",
  prompt_ids: [104, 105],
  created: 0,
  nodes: [node(0, null)],
};

function step(bytes = 100): StepData {
  return { bytes };
}

function freshWithChain(len: number): Store {
  const s = new Store();
  s.resetSession(sess);
  for (let i = 1; i <= len; i++) s.applyToken(node(i, i - 1), step(), i * 50);
  return s;
}

describe("session + token stream", () => {
  it("resetSession indexes the nodes and clears the rest", () => {
    const s = freshWithChain(0);
    expect(s.state.session?.id).toBe("abc");
    expect(s.state.nodes[0].parent).toBeNull();
    expect(s.state.tip).toBe(0);
    expect(s.state.path).toEqual([]);
  });

  it("tokens grow the path and move the tip", () => {
    const s = freshWithChain(3);
    expect(s.state.tip).toBe(3);
    expect(s.state.path).toEqual([1, 2, 3]);
    expect(s.state.nodes[1].children).toEqual([2]);
  });

  it("a sibling branch switches the path without touching the old branch", () => {
    const s = freshWithChain(3);
    // fork off node 1: new child 4, then 5
    s.applyToken(node(4, 1), step(), 200);
    s.applyToken(node(5, 4), step(), 250);
    expect(s.state.path).toEqual([1, 4, 5]);
    expect(s.state.nodes[1].children).toEqual([2, 4]);
    expect(s.state.nodes[2].children).toEqual([3]); // old branch intact
  });

  it("pathTo walks parents in order", () => {
    const s = freshWithChain(3);
    expect(pathTo(s.state.nodes, 2)).toEqual([1, 2]);
    expect(pathTo(s.state.nodes, 0)).toEqual([]);
  });
});

describe("branches + checkout", () => {
  function forked(): Store {
    const s = freshWithChain(3); // trunk 1-2-3
    s.applyToken(node(4, 1), step(), 200); // fork off node 1
    s.applyToken(node(5, 4), step(), 250);
    return s;
  }

  it("branchTips finds every leaf, oldest first", () => {
    const s = forked();
    expect(branchTips(s.state.nodes)).toEqual([3, 5]);
  });

  it("a single unforked chain has one tip", () => {
    const s = freshWithChain(2);
    expect(branchTips(s.state.nodes)).toEqual([2]);
  });

  it("checkout switches tip and path without losing steps", () => {
    const s = forked();
    expect(s.state.path).toEqual([1, 4, 5]);
    s.checkout(3);
    expect(s.state.tip).toBe(3);
    expect(s.state.path).toEqual([1, 2, 3]);
    expect(s.state.steps[5]).toBeDefined(); // the other timeline is still there
  });

  it("checkout clears selection so focus follows the new timeline", () => {
    const s = forked();
    s.select(5);
    s.checkout(3);
    expect(s.state.selected).toBeNull();
    expect(focusNode(s.state)).toBe(3);
  });

  it("checkout of an unknown node is a no-op", () => {
    const s = forked();
    s.checkout(999);
    expect(s.state.tip).toBe(5);
  });
});

describe("stats", () => {
  it("bytes per token is a running average", () => {
    const s = freshWithChain(0);
    s.applyToken(node(1, 0), step(100), 0);
    s.applyToken(node(2, 1), step(300), 50);
    expect(s.state.stats.tokens).toBe(2);
    expect(s.state.stats.bytesPerTok).toBe(200);
  });

  it("tok/s comes from inter-token gaps, not wall time of the first token", () => {
    const s = freshWithChain(0);
    s.applyToken(node(1, 0), step(), 1000);
    expect(s.state.stats.tokPerSec).toBe(0); // one token, no gap yet
    s.applyToken(node(2, 1), step(), 1100); // 100ms gap -> 10 tok/s
    expect(s.state.stats.tokPerSec).toBeCloseTo(10, 5);
  });

  it("startGeneration resets the gap clock so pauses don't poison the ema", () => {
    const s = freshWithChain(0);
    s.applyToken(node(1, 0), step(), 0);
    s.applyToken(node(2, 1), step(), 100);
    s.startGeneration();
    // a long think-pause before the user hit generate again
    s.applyToken(node(3, 2), step(), 60_000);
    s.applyToken(node(4, 3), step(), 60_100);
    expect(s.state.stats.tokPerSec).toBeGreaterThan(5);
  });
});

describe("captures", () => {
  it("merges scrub results into the step and keeps original bytes", () => {
    const s = freshWithChain(2);
    s.applyCapture(1, { bytes: 999, residNorms: new Float32Array([1, 2, 3]) });
    expect(s.state.steps[1].residNorms).toHaveLength(3);
    expect(s.state.steps[1].bytes).toBe(100); // stream bytes, not scrub bytes
  });

  it("a full-square attn frame becomes the heatmap matrix", () => {
    const s = freshWithChain(2);
    s.applyCapture(2, {
      bytes: 0,
      attn: [{ layer: 3, heads: 2, q: 4, kv: 4, data: new Float32Array(32) }],
    });
    expect(s.state.attnMatrix).toMatchObject({ forNode: 2, layer: 3, heads: 2, n: 4 });
  });

  it("a streaming-shaped attn row (q=1) does not become the matrix", () => {
    const s = freshWithChain(2);
    s.applyCapture(2, {
      bytes: 0,
      attn: [{ layer: 0, heads: 2, q: 1, kv: 4, data: new Float32Array(8) }],
    });
    expect(s.state.attnMatrix).toBeNull();
  });
});

describe("compare sides", () => {
  const sessA: SessionDict = { ...sess, id: "cmp-a", prompt: "a", prompt_ids: [104] };
  const sessB: SessionDict = { ...sess, id: "cmp-b", prompt: "b", prompt_ids: [105, 106] };

  function twoSides(): Store {
    const s = new Store();
    s.initCompareSide("a", sessA);
    s.initCompareSide("b", sessB);
    return s;
  }

  it("routes a capture reply to the side its session belongs to", () => {
    const s = twoSides();
    expect(s.compareSideOf("cmp-a")).toBe("a");
    expect(s.compareSideOf("cmp-b")).toBe("b");
    expect(s.compareSideOf("")).toBeNull();
    expect(s.compareSideOf("some other session")).toBeNull();
  });

  it("one capture fills the square, the candidates and the lens at once", () => {
    const s = twoSides();
    const lens = {
      layers: 2,
      k: 2,
      ids: new Uint32Array([7, 8, 9, 10]),
      probs: new Float32Array([0.4, 0.3, 0.2, 0.1]),
    };
    s.applyCompareCapture("a", 0, {
      bytes: 0,
      topkIds: new Uint32Array([7, 8]),
      topkLogits: new Float32Array([2, 1]),
      lens,
      residNorms: new Float32Array([1, 2, 3]),
      attn: [{ layer: 1, heads: 2, q: 3, kv: 3, data: new Float32Array(18) }],
    });
    const a = s.state.compare.a!;
    expect(a.attnMatrix).toMatchObject({ forNode: 0, layer: 1, heads: 2, n: 3 });
    expect(Array.from(a.topkIds!)).toEqual([7, 8]);
    expect(a.lens).toBe(lens);
    expect(Array.from(a.residNorms!)).toEqual([1, 2, 3]);
    expect(a.capturePending).toBe(false);
    // the other column is untouched
    expect(s.state.compare.b!.topkIds).toBeNull();
  });

  it("a layer change drops the squares but keeps the layer-independent capture", () => {
    const s = twoSides();
    s.applyCompareCapture("a", 0, {
      bytes: 0,
      topkIds: new Uint32Array([7]),
      topkLogits: new Float32Array([2]),
      attn: [{ layer: 0, heads: 1, q: 2, kv: 2, data: new Float32Array(4) }],
    });
    expect(s.state.compare.a!.attnMatrix).not.toBeNull();
    s.setCompareAttn(2, -1);
    expect(s.state.compare.a!.attnMatrix).toBeNull();
    expect(s.state.compare.a!.topkIds).not.toBeNull();
  });

  it("turning attention off clears both squares", () => {
    const s = twoSides();
    s.setCompareAttnEnabled(false);
    expect(s.state.compare.attn).toBe(false);
    s.setCompareAttnEnabled(true);
    expect(s.state.compare.attn).toBe(true);
  });

  it("a per-side failure is reported on that side only", () => {
    const s = twoSides();
    s.setComparePending("a", true);
    s.setCompareError("a", "model unloaded");
    expect(s.state.compare.a!.error).toBe("model unloaded");
    expect(s.state.compare.a!.capturePending).toBe(false);
    expect(s.state.compare.b!.error).toBeNull();
  });
});

describe("focusNode", () => {
  it("hover beats selection beats tip", () => {
    const s = freshWithChain(3);
    expect(focusNode(s.state)).toBe(3);
    s.select(1);
    expect(focusNode(s.state)).toBe(1);
    s.hover(2);
    expect(focusNode(s.state)).toBe(2);
    s.hover(null);
    expect(focusNode(s.state)).toBe(1);
  });

  it("empty session focuses nothing", () => {
    const s = freshWithChain(0);
    expect(focusNode(s.state)).toBeNull();
  });
});
