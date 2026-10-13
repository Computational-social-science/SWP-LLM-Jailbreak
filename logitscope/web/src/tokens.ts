// token id -> text, lazily batch-decoded by the server. frames carry ids only
// (u32s are cheap, strings are not), so views ask for text as they need it and
// a microbatch flushes every few ms.

import { store } from "./store";

const pending = new Set<number>();
let timer: ReturnType<typeof setTimeout> | null = null;

export function requestTexts(ids: Iterable<number>): void {
  const have = store.state.tokenTexts;
  for (const id of ids) if (!(id in have)) pending.add(id);
  if (pending.size === 0 || timer !== null) return;
  timer = setTimeout(() => void flush(), 25);
}

async function flush(): Promise<void> {
  timer = null;
  const model = store.state.session?.model ?? store.state.modelName;
  const ids = [...pending];
  pending.clear();
  if (!model || ids.length === 0) return;
  try {
    const res = await fetch(`/api/models/${model}/decode`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ ids }),
    });
    if (!res.ok) return; // model got unloaded mid-flight; views show raw ids
    const { texts } = (await res.json()) as { texts: string[] };
    const map: Record<number, string> = {};
    ids.forEach((id, i) => (map[id] = texts[i]));
    store.mergeTexts(map);
  } catch {
    // decode is cosmetic — never let it take the app down
  }
}

/** printable form: whitespace made visible so " the" and "the" look different. */
export function visible(text: string): string {
  return text.replace(/ /g, "␣").replace(/\n/g, "⏎").replace(/\t/g, "⇥");
}

/** same, for the compare view's character diff: every substitution stays one
 * character wide, so a diff computed on the printed form lines up with the
 * indices of the raw text. */
export function visibleForDiff(text: string): string {
  return text.replace(/\n/g, "⏎").replace(/\t/g, "⇥").replace(/\u200b/g, "·");
}
