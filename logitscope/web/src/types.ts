// mirrors of the server's json shapes. keep field names snake_case where the
// wire uses snake_case — renaming at the boundary just invites drift.

export interface NodeDict {
  id: number;
  parent: number | null;
  token_id: number | null;
  text: string;
  logprob: number | null;
  rank: number | null;
  entropy: number | null;
  children: number[];
}

export interface SessionDict {
  id: string;
  model: string;
  prompt: string;
  prompt_ids: number[];
  created: number;
  nodes: NodeDict[];
}

export interface ModelInfo {
  name: string;
  hf_id: string;
  loaded: boolean;
  n_layers?: number;
  n_heads?: number;
  device?: string;
}

export interface LoadProgress {
  phase: "download" | "loading" | "ready" | "error";
  done_bytes: number;
  total_bytes: number;
  error?: string;
}

export interface SamplingParams {
  temperature: number;
  top_k: number;
  top_p: number;
  repetition_penalty: number;
  seed: number;
  max_new_tokens: number;
}

export interface CaptureConfig {
  topk_logits?: number;
  hidden?: boolean;
  resid_norms?: boolean;
  lens?: boolean;
  lens_topk?: number;
  attn_layers?: number[];
  attn_heads?: number[];
}

export const ROOT = 0; // node 0 anchors the prompt, same as the server

export const DEFAULT_PARAMS: SamplingParams = {
  temperature: 0.8,
  top_k: 0,
  top_p: 1.0,
  repetition_penalty: 1.0,
  seed: 0,
  max_new_tokens: 48,
};
