import { useState } from "react";
import type { LogitLensResult } from "../api";
import { color, font } from "../theme";
import { LogitLensGrid } from "./LogitLensGrid";
import { LogitLensStream } from "./LogitLensStream";
import { EmptyState } from "./EmptyState";
import { LoadingOverlay } from "./LoadingOverlay";
import { Explainer } from "./Explainer";

type Mode = "grid" | "stream";

// Shown at the top of the tab, before and after a run — mirrors Attention/Ablation.
function ModeToggle({ mode, setMode }: { mode: Mode; setMode: (m: Mode) => void }) {
  return (
    <div className="flex rounded-md overflow-hidden" style={{ border: `1px solid ${color.border}` }}>
      {(["stream", "grid"] as const).map((m) => (
        <button
          key={m}
          onClick={() => setMode(m)}
          className="gb-segbtn px-3 py-1 text-xs"
          style={{
            fontFamily: font.ui,
            backgroundColor: mode === m ? color.accent : color.surface,
            color: mode === m ? color.bg : color.textMd,
            fontWeight: mode === m ? 700 : 400,
          }}
        >
          {m}
        </button>
      ))}
    </div>
  );
}

function LensExplainer() {
  return (
    <Explainer title="What am I looking at?">
      <p>
        At every layer we take the model's half-finished thought (the{" "}
        <strong>residual stream</strong>) and decode it through the model's own output head — asking
        "what would it predict if it had to answer <em>right here</em>?" Each row is one layer's
        answer, bottom (raw word lookup) to top (the real prediction).
      </p>
      <p>
        Colour and bar track the probability of the final answer at each height: dim is noise, bright
        is locked in. The dashed gold rule marks the layer where the answer first wins. Sublabels
        read <code>N_pre</code> = before layer N runs, <code>final_post</code> = after the last layer.
      </p>
      <p style={{ color: color.textLo }}>
        Why it's useful: you can watch a prediction assemble. Often the answer is undecided for the
        first half of the network, snaps into place in a few middle layers, then just sharpens — the
        logit lens is how researchers first saw that "thinking" happen layer by layer.
      </p>
    </Explainer>
  );
}

export function LogitLensSection({
  result,
  busy,
  error,
  simulating = false,
  genFrom,
  delayMs = 0,
  animate = true,
}: {
  result: LogitLensResult | null;
  busy: boolean;
  error: string | null;
  simulating?: boolean;
  genFrom?: number;
  delayMs?: number;
  animate?: boolean;
}) {
  const [mode, setMode] = useState<Mode>("stream");

  if (!result) {
    return (
      <div className="space-y-3">
        <LensExplainer />
        <div className="flex justify-end">
          <ModeToggle mode={mode} setMode={setMode} />
        </div>
        {busy || error ? null : <EmptyState />}
      </div>
    );
  }

  return (
    // key on model only — a Simulate run streams many results for the same model, so this stays
    // mounted and updates in place instead of remounting (and re-flashing) every token.
    <div className="gb-fade-up space-y-3 relative" key={result.model_name}>
      <LensExplainer />
      <div className="flex flex-wrap items-center gap-3">
        <p className="text-sm" style={{ fontFamily: font.ui, color: color.textMd }}>
          {simulating ? "generating:" : "final prediction:"}{" "}
          <span style={{ fontFamily: font.mono, color: color.accent, fontWeight: 500 }}>
            {result.final_top_token.trim() || "·"}
          </span>
          {mode === "grid" && (
            <span style={{ color: color.textLo }}> · click a token column to see its journey in stream mode</span>
          )}
          {mode === "stream" && genFrom != null && (
            <span style={{ color: color.textLo }}> · underlined tokens were generated — click one to see its journey</span>
          )}
        </p>
        <div className="ml-auto">
          <ModeToggle mode={mode} setMode={setMode} />
        </div>
      </div>

      {mode === "grid" ? (
        <LogitLensGrid result={result} genFrom={genFrom} />
      ) : (
        <LogitLensStream result={result} genFrom={genFrom} simulating={simulating} delayMs={delayMs} animate={animate} />
      )}

      <LoadingOverlay active={busy} />
    </div>
  );
}
