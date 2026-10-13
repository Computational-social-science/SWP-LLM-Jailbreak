import { describe, expect, it } from "vitest";
import { buildStep } from "./client";
import type { Frame } from "./wire";

function f32(shape: number[], vals: number[]): Frame {
  return { kind: 0, shape, data: new Float32Array(vals) };
}

function u32(shape: number[], vals: number[]): Frame {
  return { kind: 0, shape, data: new Uint32Array(vals) };
}

describe("buildStep", () => {
  it("assembles topk + lens + resid from the meta/frame pairs", () => {
    const step = buildStep(
      [
        { kind: "topk_ids" },
        { kind: "topk_logits" },
        { kind: "lens_ids" },
        { kind: "lens_probs" },
        { kind: "resid_norms" },
      ],
      [
        u32([2], [7, 9]),
        f32([2], [1.5, -2]),
        u32([3, 2], [1, 2, 3, 4, 5, 6]),
        f32([3, 2], [0.9, 0.1, 0.8, 0.2, 0.7, 0.3]),
        f32([3], [1, 2, 3]),
      ],
      1234,
    );
    expect(Array.from(step.topkIds!)).toEqual([7, 9]);
    expect(step.lens).toMatchObject({ layers: 3, k: 2 });
    expect(step.lens!.probs[0]).toBeCloseTo(0.9);
    expect(step.residNorms).toHaveLength(3);
    expect(step.bytes).toBe(1234);
  });

  it("collects attn blocks with their layer from the meta", () => {
    const attn: Frame = { kind: 5, shape: [2, 4, 4], data: new Float32Array(32) };
    const step = buildStep([{ kind: "attn", layer: 11 }], [attn], 0);
    expect(step.attn).toHaveLength(1);
    expect(step.attn![0]).toMatchObject({ layer: 11, heads: 2, q: 4, kv: 4 });
  });

  it("rejects lens_probs arriving before lens_ids", () => {
    expect(() =>
      buildStep([{ kind: "lens_probs" }], [f32([1, 1], [0.5])], 0),
    ).toThrow(/before lens_ids/);
  });

  it("rejects unknown kinds so protocol drift is loud", () => {
    expect(() => buildStep([{ kind: "wat" }], [f32([1], [0])], 0)).toThrow(/unknown frame kind/);
  });
});
