import type { ModelInfo } from "../api";
import { color, font } from "../theme";

// Text input + model picker + Run button. No <form> — plain handlers, per the plan.
export function PromptBar({
  prompt,
  setPrompt,
  models,
  model,
  setModel,
  onRun,
  busy,
}: {
  prompt: string;
  setPrompt: (v: string) => void;
  models: ModelInfo[];
  model: string;
  setModel: (v: string) => void;
  onRun: () => void;
  busy: boolean;
}) {
  return (
    <div className="flex gap-2 items-stretch">
      <input
        className="gb-input flex-1 rounded-md px-3 py-2 text-sm outline-none"
        style={{
          fontFamily: font.ui,
          backgroundColor: color.surface,
          border: `1px solid ${color.border}`,
          color: color.textHi,
        }}
        placeholder="Type a prompt, e.g. The capital of France is"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !busy) onRun();
        }}
      />
      <select
        className="gb-select rounded-md px-2 py-2 text-sm outline-none"
        style={{
          fontFamily: font.mono,
          backgroundColor: color.surface,
          border: `1px solid ${color.border}`,
          color: color.textHi,
        }}
        value={model}
        onChange={(e) => setModel(e.target.value)}
      >
        {models.map((m) => (
          <option key={m.name} value={m.name} disabled={m.gated && !m.loaded}>
            {m.display_name}
            {m.gated && !m.loaded ? " (locked)" : ""}
          </option>
        ))}
      </select>
      <button
        className="gb-btn rounded-md px-4 py-2 text-sm font-medium disabled:opacity-50"
        style={{ fontFamily: font.ui, backgroundColor: color.accent, color: color.bg }}
        onClick={onRun}
        disabled={busy || prompt.trim().length === 0}
      >
        {busy ? "Running…" : "Run"}
      </button>
    </div>
  );
}
