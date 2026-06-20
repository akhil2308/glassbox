import type { ModelInfo } from "../api";
import { color, font } from "../theme";
import { Select } from "./Select";

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
    <div className="flex flex-col sm:flex-row gap-2 items-stretch">
      <input
        className="gb-input flex-1 rounded-md px-3 py-2 text-sm outline-none min-w-0"
        style={{
          fontFamily: font.ui,
          backgroundColor: color.surface,
          border: `1px solid ${color.border}`,
          color: color.textHi,
        }}
        placeholder="Type a prompt, e.g. The capital of France is"
        aria-label="Prompt"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !busy) onRun();
        }}
      />
      <div className="flex gap-2 items-stretch">
        <Select
          className="flex-1 sm:flex-initial sm:w-56 text-sm"
          width="100%"
          mono
          ariaLabel="Model"
          value={model}
          onChange={setModel}
          options={models.map((m) => ({
            value: m.name,
            label: m.display_name + (m.gated && !m.loaded ? " (locked)" : ""),
            disabled: m.gated && !m.loaded,
          }))}
        />
        <button
          className="gb-btn shrink-0 rounded-md px-4 py-2 text-sm font-semibold disabled:opacity-50"
          style={{ fontFamily: font.ui, backgroundColor: color.accent, color: color.bg }}
          aria-label="Run forward pass"
          onClick={onRun}
          disabled={busy || prompt.trim().length === 0}
        >
          {busy ? (
            <span className="inline-flex items-center gap-1" aria-hidden="true">
              {[0, 1, 2].map((i) => (
                <span
                  key={i}
                  className="gb-pulse-dot inline-block rounded-full"
                  style={{
                    width: 5,
                    height: 5,
                    backgroundColor: color.bg,
                    animationDelay: `${i * 0.15}s`,
                  }}
                />
              ))}
            </span>
          ) : (
            "Run"
          )}
        </button>
      </div>
    </div>
  );
}
