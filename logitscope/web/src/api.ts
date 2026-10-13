// thin rest client. everything important streams over the websocket; this is
// just models and session bookkeeping.

import type { LoadProgress, ModelInfo, SessionDict } from "./types";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = ((await res.json()) as { detail?: string }).detail ?? detail;
    } catch {
      // non-json error body, statusText is fine
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export async function listModels(): Promise<ModelInfo[]> {
  return json(await fetch("/api/models"));
}

export async function loadModel(name: string): Promise<{ name: string; device: string }> {
  return json(
    await fetch("/api/models/load", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name }),
    }),
  );
}

export async function loadProgress(): Promise<Record<string, LoadProgress>> {
  return json(await fetch("/api/models/progress"));
}

export async function createSession(model: string, prompt: string): Promise<SessionDict> {
  return json(
    await fetch("/api/sessions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model, prompt }),
    }),
  );
}

export async function getSession(id: string): Promise<SessionDict> {
  return json(await fetch(`/api/sessions/${id}`));
}

export async function exportSession(id: string): Promise<string> {
  const res = await fetch(`/api/sessions/${id}/export`);
  if (!res.ok) throw new Error(`${res.status}: export failed`);
  return res.text();
}

export async function importSession(text: string): Promise<SessionDict> {
  return json(await fetch("/api/sessions/import", { method: "POST", body: text }));
}
