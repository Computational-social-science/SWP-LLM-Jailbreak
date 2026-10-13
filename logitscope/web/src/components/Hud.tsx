// the front panel: model picker, sampling knobs (applied live to the next
// tokens), capture toggles, and the little meters that make it feel like an
// instrument.

import { loadModel } from "../actions";
import { store, useStore } from "../store";
import type { LoadProgress } from "../types";

function loadLabel(p: LoadProgress | null): string {
  if (p?.phase === "download" && p.total_bytes > 0) {
    // hold at 99 until the server flips phase — the poll lags the truth
    const pct = Math.min(99, Math.round((100 * p.done_bytes) / p.total_bytes));
    return `downloading ${pct}%`;
  }
  if (p?.phase === "loading") return "warming up…";
  return "loading…";
}

function Knob({
  label,
  value,
  min,
  max,
  step,
  fmt,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  fmt: (v: number) => string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="knob" title={label}>
      <span className="knob-label">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <span className="knob-value">{fmt(value)}</span>
    </label>
  );
}

export default function Hud() {
  const conn = useStore((s) => s.conn);
  const models = useStore((s) => s.models);
  const modelName = useStore((s) => s.modelName);
  const modelLoading = useStore((s) => s.modelLoading);
  const progress = useStore((s) => s.loadProgress);
  const params = useStore((s) => s.params);
  const capture = useStore((s) => s.capture);
  const stats = useStore((s) => s.stats);
  const generating = useStore((s) => s.generating);
  const compareMode = useStore((s) => s.compareMode);

  const selected = models.find((m) => m.name === modelName);

  return (
    <header className="hud">
      <span className="wordmark">logitscope</span>
      <span className={`conn conn-${conn}`} title="websocket">
        ●
      </span>

      <button
        className={compareMode ? "compare-toggle compare-toggle-on" : "compare-toggle"}
        onClick={() => store.set({ compareMode: !compareMode })}
      >
        compare
      </button>

      <span className="hud-group">
        <select
          value={modelName}
          disabled={modelLoading}
          onChange={(e) => store.set({ modelName: e.target.value })}
        >
          {models.map((m) => (
            <option key={m.name} value={m.name}>
              {m.name}
              {m.loaded ? ` (${m.device})` : ""}
            </option>
          ))}
        </select>
        {selected && !selected.loaded && (
          <button disabled={modelLoading} onClick={() => void loadModel(modelName)}>
            {modelLoading ? loadLabel(progress) : "load"}
          </button>
        )}
      </span>

      <span className="hud-group hud-knobs">
        <Knob
          label="temp"
          value={params.temperature}
          min={0}
          max={2}
          step={0.05}
          fmt={(v) => v.toFixed(2)}
          onChange={(v) => store.setParams({ temperature: v })}
        />
        <Knob
          label="top-k"
          value={params.top_k}
          min={0}
          max={64}
          step={1}
          fmt={(v) => (v === 0 ? "off" : String(v))}
          onChange={(v) => store.setParams({ top_k: v })}
        />
        <Knob
          label="top-p"
          value={params.top_p}
          min={0.05}
          max={1}
          step={0.01}
          fmt={(v) => (v === 1 ? "off" : v.toFixed(2))}
          onChange={(v) => store.setParams({ top_p: v })}
        />
        <label className="knob" title="seed">
          <span className="knob-label">seed</span>
          <input
            className="hud-num"
            type="number"
            value={params.seed}
            onChange={(e) => store.setParams({ seed: Number(e.target.value) || 0 })}
          />
        </label>
        <label className="knob" title="tokens per generate">
          <span className="knob-label">tokens</span>
          <input
            className="hud-num"
            type="number"
            min={1}
            max={256}
            value={params.max_new_tokens}
            onChange={(e) =>
              store.setParams({
                max_new_tokens: Math.max(1, Math.min(256, Number(e.target.value) || 1)),
              })
            }
          />
        </label>
      </span>

      <span className="hud-group">
        <label className="toggle">
          <input
            type="checkbox"
            checked={capture.lens}
            onChange={(e) => store.setCapture({ lens: e.target.checked })}
          />
          lens
        </label>
        <label className="toggle">
          <input
            type="checkbox"
            checked={capture.residNorms}
            onChange={(e) => store.setCapture({ residNorms: e.target.checked })}
          />
          resid
        </label>
      </span>

      <span className="hud-spacer" />

      <span className="meter" title="decode speed (ema of inter-token gaps)">
        <span className={generating ? "meter-value meter-live" : "meter-value"}>
          {stats.tokPerSec > 0 ? stats.tokPerSec.toFixed(1) : "–"}
        </span>{" "}
        tok/s
      </span>
      <span className="meter" title="wire bytes per token, json + binary frames">
        <span className="meter-value">
          {stats.bytesPerTok > 0 ? (stats.bytesPerTok / 1024).toFixed(1) : "–"}
        </span>{" "}
        KiB/tok
      </span>
    </header>
  );
}
