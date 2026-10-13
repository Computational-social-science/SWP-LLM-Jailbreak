// websocket client. the server speaks: one json "token" (or "capture") message
// announcing N frames, then N binary frames, then eventually "done".
//
// the trap: ws delivers messages synchronously in order, but our frame decode
// is async (DecompressionStream). without care, frame k+1 can finish inflating
// before frame k, or the next token's json can be handled while this token's
// frames are still in flight. every incoming message goes through one promise
// chain so nothing is processed out of order.

import { KIND_ATTN, decodeFrame, type Frame } from "./wire";
import type { CaptureConfig, NodeDict, SamplingParams } from "./types";

export interface AttnBlock {
  layer: number;
  heads: number;
  q: number;
  kv: number;
  data: Float32Array;
}

export interface StepData {
  topkIds?: Uint32Array;
  topkLogits?: Float32Array;
  lens?: { layers: number; k: number; ids: Uint32Array; probs: Float32Array };
  residNorms?: Float32Array;
  attn?: AttnBlock[];
  bytes: number; // wire bytes for this step, json + binary
}

interface FrameMeta {
  kind: string;
  layer?: number;
}

export interface SocketEvents {
  onToken?(sessionId: string, node: NodeDict, step: StepData): void;
  onCapture?(sessionId: string, nodeId: number, cached: boolean, step: StepData): void;
  onDone?(sessionId: string, tokens: number): void;
  onError?(sessionId: string | null, err: string): void;
  onStatus?(status: "open" | "closed"): void;
}

interface Pending {
  type: "token" | "capture";
  sessionId: string;
  meta: FrameMeta[];
  frames: Frame[];
  bytes: number;
  node?: NodeDict;
  nodeId?: number;
  cached?: boolean;
}

export function buildStep(meta: FrameMeta[], frames: Frame[], bytes: number): StepData {
  const step: StepData = { bytes };
  for (let i = 0; i < meta.length; i++) {
    const m = meta[i];
    const f = frames[i];
    switch (m.kind) {
      case "topk_ids":
        step.topkIds = f.data as Uint32Array;
        break;
      case "topk_logits":
        step.topkLogits = f.data as Float32Array;
        break;
      case "lens_ids":
        step.lens = {
          layers: f.shape[0],
          k: f.shape[1],
          ids: f.data as Uint32Array,
          probs: step.lens?.probs ?? new Float32Array(0),
        };
        break;
      case "lens_probs":
        if (!step.lens) throw new Error("lens_probs frame before lens_ids");
        step.lens.probs = f.data as Float32Array;
        break;
      case "resid_norms":
        step.residNorms = f.data as Float32Array;
        break;
      case "attn": {
        if (f.kind !== KIND_ATTN || f.shape.length !== 3)
          throw new Error(`attn frame with shape [${f.shape}]`);
        const block: AttnBlock = {
          layer: m.layer ?? 0,
          heads: f.shape[0],
          q: f.shape[1],
          kv: f.shape[2],
          data: f.data as Float32Array,
        };
        (step.attn ??= []).push(block);
        break;
      }
      case "hidden":
        break; // we never ask for raw hidden states in the ui
      default:
        throw new Error(`unknown frame kind ${m.kind}`);
    }
  }
  return step;
}

export class ScopeSocket {
  private ws: WebSocket | null = null;
  private chain: Promise<void> = Promise.resolve();
  private pending: Pending | null = null;

  constructor(private events: SocketEvents) {}

  connect(url = `ws://${location.host}/ws`): void {
    this.detach(); // react strict-mode double-mounts: don't let the socket
    // we're replacing fire a scary "websocket error" into the ui
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => this.events.onStatus?.("open");
    ws.onclose = () => this.events.onStatus?.("closed");
    ws.onerror = () => this.events.onError?.(null, "websocket error");
    ws.onmessage = (ev) => this.enqueue(() => this.handle(ev.data as string | ArrayBuffer));
    this.ws = ws;
  }

  close(): void {
    this.detach();
  }

  private detach(): void {
    if (!this.ws) return;
    this.ws.onopen = this.ws.onclose = this.ws.onerror = this.ws.onmessage = null;
    this.ws.close();
    this.ws = null;
  }

  generate(sessionId: string, fromNode: number, params: SamplingParams, capture: CaptureConfig): void {
    this.send({ type: "generate", session_id: sessionId, from_node: fromNode, params, capture });
  }

  /** "what if it had said X" — the server replaces node_id's token with
   * token_id as a sibling branch and keeps generating from there. the reply
   * is ordinary token messages, so the stream handlers don't care. */
  fork(sessionId: string, nodeId: number, tokenId: number, params: SamplingParams, capture: CaptureConfig): void {
    this.send({ type: "fork", session_id: sessionId, node_id: nodeId, token_id: tokenId, params, capture });
  }

  capture(sessionId: string, nodeId: number, capture: CaptureConfig): void {
    this.send({ type: "capture", session_id: sessionId, node_id: nodeId, capture });
  }

  /** attention over the prompt tokens only — no generation, no node. */
  capturePrompt(sessionId: string, capture: CaptureConfig): void {
    this.send({ type: "capture_prompt", session_id: sessionId, capture });
  }

  private send(msg: object): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.events.onError?.(null, "not connected");
      return;
    }
    this.ws.send(JSON.stringify(msg));
  }

  private enqueue(fn: () => Promise<void>): void {
    this.chain = this.chain.then(fn).catch((e: unknown) => {
      this.pending = null;
      this.events.onError?.(null, e instanceof Error ? e.message : String(e));
    });
  }

  private async handle(data: string | ArrayBuffer): Promise<void> {
    if (typeof data !== "string") {
      const p = this.pending;
      if (!p) throw new Error("binary frame with no announcing message");
      p.frames.push(await decodeFrame(data));
      p.bytes += data.byteLength;
      if (p.frames.length === p.meta.length) this.flush(p);
      return;
    }

    const msg = JSON.parse(data) as Record<string, unknown>;
    switch (msg.type) {
      case "token": {
        this.pending = {
          type: "token",
          sessionId: msg.session_id as string,
          meta: msg.frames as FrameMeta[],
          frames: [],
          bytes: data.length,
          node: msg.node as NodeDict,
        };
        break;
      }
      case "capture": {
        this.pending = {
          type: "capture",
          sessionId: msg.session_id as string,
          meta: msg.frames as FrameMeta[],
          frames: [],
          bytes: data.length,
          nodeId: msg.node_id as number,
          cached: msg.cached as boolean,
        };
        break;
      }
      case "done":
        this.events.onDone?.(msg.session_id as string, msg.tokens as number);
        return;
      case "error":
        this.events.onError?.((msg.session_id as string | null) ?? null, msg.error as string);
        return;
      default:
        throw new Error(`unknown message type ${String(msg.type)}`);
    }
    // a capture-free token has zero frames and flushes immediately
    if (this.pending.meta.length === 0) this.flush(this.pending);
  }

  private flush(p: Pending): void {
    this.pending = null;
    const step = buildStep(p.meta, p.frames, p.bytes);
    if (p.type === "token") this.events.onToken?.(p.sessionId, p.node!, step);
    else this.events.onCapture?.(p.sessionId, p.nodeId!, p.cached!, step);
  }
}
