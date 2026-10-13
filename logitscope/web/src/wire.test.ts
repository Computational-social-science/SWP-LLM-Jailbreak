import { describe, expect, it } from "vitest";
import { decodeFrame, f16ToF32 } from "./wire";
import golden from "./wire.golden.json";

// -- f16, known bit patterns --

describe("f16ToF32", () => {
  const cases: [number, number][] = [
    [0x0000, 0],
    [0x3c00, 1],
    [0xc000, -2],
    [0x3555, 0.333251953125], // closest f16 to 1/3
    [0x7bff, 65504], // f16 max
    [0x0400, 6.103515625e-5], // smallest normal
    [0x0001, 2 ** -24], // smallest subnormal
    [0x7c00, Infinity],
    [0xfc00, -Infinity],
  ];
  it.each(cases)("0x%s", (bits, want) => {
    expect(f16ToF32(bits)).toBe(want);
  });

  it("negative zero", () => {
    expect(Object.is(f16ToF32(0x8000), -0)).toBe(true);
  });

  it("nan", () => {
    expect(f16ToF32(0x7e00)).toBeNaN();
  });
});

// -- golden frames straight out of the python encoder --

function b64ToBuf(b64: string): ArrayBuffer {
  const s = atob(b64);
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out.buffer;
}

function thaw(v: number | string): number {
  // json can't hold nan/inf, the generator ships them as strings
  if (v === "NaN") return NaN;
  if (v === "Infinity") return Infinity;
  if (v === "-Infinity") return -Infinity;
  return v as number;
}

describe("golden frames from python", () => {
  it.each(golden.map((c) => [c.name, c] as const))("%s", async (_, c) => {
    const frame = await decodeFrame(b64ToBuf(c.b64));
    expect(frame.kind).toBe(c.kind);
    expect(frame.shape).toEqual(c.shape);
    expect(frame.data.length).toBe(c.expected.length);
    for (let i = 0; i < c.expected.length; i++) {
      const want = thaw(c.expected[i]);
      if (Number.isNaN(want)) expect(frame.data[i]).toBeNaN();
      else expect(frame.data[i]).toBe(want); // exact: expected is f16-quantized
    }
  });
});

// -- hand-built frames for the error paths --

async function gzip(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([bytes as BlobPart])
    .stream()
    .pipeThrough(new CompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function buildFrame(opts: {
  magic?: [number, number];
  version?: number;
  kind?: number;
  dtype?: number;
  shape?: number[];
  payload: Uint8Array;
}): Promise<ArrayBuffer> {
  const [m0, m1] = opts.magic ?? [0x4c, 0x53];
  const shape = opts.shape ?? [];
  const zipped = await gzip(opts.payload);
  const buf = new Uint8Array(6 + 4 * shape.length + zipped.length);
  const view = new DataView(buf.buffer);
  buf[0] = m0;
  buf[1] = m1;
  buf[2] = opts.version ?? 1;
  buf[3] = opts.kind ?? 2;
  buf[4] = opts.dtype ?? 3;
  buf[5] = shape.length;
  shape.forEach((s, i) => view.setUint32(6 + 4 * i, s, true));
  buf.set(zipped, 6 + 4 * shape.length);
  return buf.buffer;
}

describe("decode errors", () => {
  const u32payload = (...vals: number[]) => new Uint8Array(new Uint32Array(vals).buffer);

  it("roundtrips a js-built u32 frame", async () => {
    const buf = await buildFrame({ shape: [3], payload: u32payload(7, 8, 9) });
    const frame = await decodeFrame(buf);
    expect(Array.from(frame.data)).toEqual([7, 8, 9]);
  });

  it("rejects bad magic", async () => {
    const buf = await buildFrame({ magic: [0x58, 0x58], shape: [1], payload: u32payload(1) });
    await expect(decodeFrame(buf)).rejects.toThrow(/magic/);
  });

  it("rejects unknown version", async () => {
    const buf = await buildFrame({ version: 9, shape: [1], payload: u32payload(1) });
    await expect(decodeFrame(buf)).rejects.toThrow(/version/);
  });

  it("rejects unknown dtype", async () => {
    const buf = await buildFrame({ dtype: 42, shape: [1], payload: u32payload(1) });
    await expect(decodeFrame(buf)).rejects.toThrow(/dtype/);
  });

  it("rejects payload/shape mismatch", async () => {
    const buf = await buildFrame({ shape: [5], payload: u32payload(1, 2) });
    await expect(decodeFrame(buf)).rejects.toThrow(/shape says/);
  });

  it("rejects a frame truncated inside the shape", async () => {
    const buf = new Uint8Array([0x4c, 0x53, 1, 2, 3, 2, 0, 0]); // ndim=2, one u32 missing
    await expect(decodeFrame(buf.buffer)).rejects.toThrow(/truncated/);
  });
});
