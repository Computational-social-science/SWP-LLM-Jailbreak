// binary ws frame decoder — the js mirror of server/logitscope/wire.py.
//
//   "LS" | version u8 | kind u8 | dtype u8 | ndim u8 | shape u32 x ndim | gzip(data)
//
// everything little-endian. floats arrive as f16, ids as u32.

export const KIND_TOPK_IDS = 1;
export const KIND_TOPK_LOGITS = 2;
export const KIND_LENS_IDS = 3;
export const KIND_LENS_PROBS = 4;
export const KIND_ATTN = 5;
export const KIND_RESID_NORMS = 6;
export const KIND_HIDDEN = 7;

const DT_F16 = 1;
const DT_F32 = 2;
const DT_U32 = 3;

const HEADER_SIZE = 6; // "LS" + version + kind + dtype + ndim

export interface Frame {
  kind: number;
  shape: number[];
  data: Float32Array | Uint32Array;
}

// no Float16Array in the wild yet, so decode by hand. f16 is
// sign(1) | exponent(5, bias 15) | fraction(10).
export function f16ToF32(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exp = (bits >> 10) & 0x1f;
  const frac = bits & 0x03ff;
  if (exp === 0) return sign * frac * 2 ** -24; // subnormal (or zero)
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

function f16ArrayToF32(bytes: Uint8Array): Float32Array {
  const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = f16ToF32(u16[i]);
  return out;
}

async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  // DecompressionStream is in every browser we care about and in node 18+,
  // so no zlib dependency
  const stream = new Blob([bytes as BlobPart])
    .stream()
    .pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export async function decodeFrame(buf: ArrayBuffer): Promise<Frame> {
  const view = new DataView(buf);
  if (buf.byteLength < HEADER_SIZE) throw new Error("frame shorter than header");
  if (view.getUint8(0) !== 0x4c || view.getUint8(1) !== 0x53)
    throw new Error("bad magic, not an LS frame");
  const version = view.getUint8(2);
  if (version !== 1) throw new Error(`unknown frame version ${version}`);
  const kind = view.getUint8(3);
  const dtype = view.getUint8(4);
  const ndim = view.getUint8(5);

  let off = HEADER_SIZE;
  if (buf.byteLength < off + 4 * ndim) throw new Error("frame truncated in shape");
  const shape: number[] = [];
  for (let i = 0; i < ndim; i++) {
    shape.push(view.getUint32(off, true));
    off += 4;
  }

  const raw = await gunzip(new Uint8Array(buf, off));
  const count = shape.reduce((a, b) => a * b, 1);

  let data: Float32Array | Uint32Array;
  let itemSize: number;
  switch (dtype) {
    case DT_F16:
      itemSize = 2;
      data = f16ArrayToF32(raw);
      break;
    case DT_F32:
      itemSize = 4;
      data = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
      break;
    case DT_U32:
      itemSize = 4;
      data = new Uint32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
      break;
    default:
      throw new Error(`unknown dtype code ${dtype}`);
  }
  if (raw.byteLength !== count * itemSize)
    throw new Error(`payload is ${raw.byteLength} bytes, shape says ${count * itemSize}`);
  return { kind, shape, data };
}
