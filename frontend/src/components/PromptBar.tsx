import type { ModelInfo } from "../api";
import { color, font } from "../theme";
import { Select } from "./Select";

// Starter prompts chosen to make each view's pattern obvious: factual recall (lens/ablation),
// the classic induction setup (attention), and a sequence the model must continue by copying.
const EXAMPLES = [
  "The capital of France is",
  "When John and Mary went to the store, John gave a drink to",
  "The Eiffel Tower is located in the city of",
  "1, 2, 3, 4,",
];

// Text input + model picker + Run button. No <form> — plain handlers, per the plan.
export function PromptBar({
  prompt,
  setPrompt,
  models,
  model,
  setModel,
  onRun,
  busy,
  showSimulate,
  onSimulate,
  onStop,
  simulating,
  maxTokens,
  setMaxTokens,
  delayMs,
  setDelayMs,
  animate,
  setAnimate,
}: {
  prompt: string;
  setPrompt: (v: string) => void;
  models: ModelInfo[];
  model: string;
  setModel: (v: string) => void;
  onRun: () => void;
  busy: boolean;
  showSimulate: boolean; // simulate + generation settings only apply to the lens tab
  onSimulate: () => void;
  onStop: () => void;
  simulating: boolean;
  maxTokens: number;
  setMaxTokens: (v: number) => void;
  delayMs: number;
  setDelayMs: (v: number) => void;
  animate: boolean;
  setAnimate: (v: boolean) => void;
}) {
  const disabled = busy || simulating || prompt.trim().length === 0;
  // Speed slider runs fast→slow left to right via delay; default 600ms is a readable pace.
  const SPEED_MAX = 1500;
  return (
   <div className="flex flex-col gap-2">
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
        readOnly={simulating}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !disabled) onRun();
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
          disabled={disabled}
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
        {showSimulate && (
          <button
            className="gb-btn shrink-0 rounded-md px-4 py-2 text-sm font-semibold disabled:opacity-50"
            style={{
              fontFamily: font.ui,
              backgroundColor: simulating ? color.danger : color.surfaceRaised,
              color: simulating ? color.bg : color.accent,
              border: `1px solid ${simulating ? color.danger : color.accent}`,
            }}
            aria-label={simulating ? "Stop generation" : "Simulate generation"}
            onClick={simulating ? onStop : onSimulate}
            disabled={!simulating && (busy || prompt.trim().length === 0)}
          >
            {simulating ? "Stop" : "Simulate →"}
          </button>
        )}
      </div>
    </div>

    {/* One-click starter prompts — fill the input so newcomers see a meaningful pattern fast. */}
    <div className="flex flex-wrap items-center gap-1.5 px-1">
      <span className="text-xs shrink-0" style={{ fontFamily: font.ui, color: color.textLo }}>
        try
      </span>
      {EXAMPLES.map((ex) => (
        <button
          key={ex}
          onClick={() => setPrompt(ex)}
          disabled={simulating}
          className="gb-btn rounded-full px-2.5 py-0.5 text-xs disabled:opacity-50 truncate max-w-[16rem]"
          style={{
            fontFamily: font.mono,
            backgroundColor: prompt === ex ? color.accent : color.surface,
            color: prompt === ex ? color.bg : color.textMd,
            border: `1px solid ${prompt === ex ? color.accent : color.border}`,
          }}
          title={ex}
        >
          {ex}
        </button>
      ))}
    </div>

    {/* Generation settings: how many tokens, how fast. Sliders avoid native spinner clutter. */}
    {showSimulate && (
    <div
      className="flex flex-wrap items-center gap-x-6 gap-y-1.5 px-1 text-xs"
      style={{ fontFamily: font.ui, color: color.textLo }}
    >
      <label className="flex items-center gap-2">
        <span className="shrink-0">simulate</span>
        <input
          type="range"
          min={1}
          max={100}
          value={maxTokens}
          disabled={simulating}
          onChange={(e) => setMaxTokens(Number(e.target.value))}
          aria-label="Tokens to generate"
          className="w-28 disabled:opacity-50"
          style={{ accentColor: color.accent }}
        />
        <span className="shrink-0 tabular-nums" style={{ fontFamily: font.mono, color: color.textMd }}>
          {maxTokens} tokens
        </span>
      </label>
      <label className="flex items-center gap-2">
        <span className="shrink-0">speed</span>
        <input
          type="range"
          min={0}
          max={SPEED_MAX}
          step={100}
          // Slider points fast→slow; invert so dragging right slows generation down.
          value={SPEED_MAX - delayMs}
          onChange={(e) => setDelayMs(SPEED_MAX - Number(e.target.value))}
          aria-label="Generation speed"
          className="w-28"
          style={{ accentColor: color.accent }}
        />
        <span className="shrink-0 tabular-nums" style={{ fontFamily: font.mono, color: color.textMd }}>
          {delayMs === 0 ? "instant" : `${(delayMs / 1000).toFixed(1)}s/tok`}
        </span>
      </label>
      <label className="flex items-center gap-2 cursor-pointer select-none">
        <input
          type="checkbox"
          checked={animate}
          onChange={(e) => setAnimate(e.target.checked)}
          className="cursor-pointer"
          style={{ accentColor: color.accent }}
        />
        <span className="shrink-0">animate layers</span>
      </label>
    </div>
    )}
   </div>
  );
}
