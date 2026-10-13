import { useRef, useState } from "react";
import { continueFrom, newSession, openSession, replaySession, saveSession } from "../actions";
import { useStore } from "../store";

export default function PromptBar() {
  const [prompt, setPrompt] = useState("");
  const generating = useStore((s) => s.generating);
  const session = useStore((s) => s.session);
  const error = useStore((s) => s.error);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const go = () => {
    if (prompt.trim() && !generating) void newSession(prompt);
  };

  return (
    <div className="prompt-bar">
      <textarea
        rows={1}
        value={prompt}
        placeholder="prompt…  (enter to run, shift+enter for newline)"
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            go();
          }
        }}
      />
      <button onClick={go} disabled={generating || !prompt.trim()}>
        generate
      </button>
      <button onClick={() => continueFrom()} disabled={generating || !session}>
        continue
      </button>
      <button
        title="download this session as jsonl"
        onClick={() => void saveSession()}
        disabled={!session}
      >
        save
      </button>
      <button
        title="load a saved session"
        onClick={() => fileRef.current?.click()}
        disabled={generating}
      >
        open
      </button>
      <button
        title="recompute captures along the current path (warms the cache)"
        onClick={() => replaySession()}
        disabled={generating || !session}
      >
        replay
      </button>
      <input
        ref={fileRef}
        type="file"
        accept=".jsonl,application/jsonl"
        style={{ display: "none" }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void openSession(f);
          e.target.value = ""; // same file twice should still fire
        }}
      />
      {error && <span className="error">{error}</span>}
    </div>
  );
}
