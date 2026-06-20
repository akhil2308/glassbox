import { useState } from "react";
import type { LogitLensResult } from "../api";
import { color, font } from "../theme";
import { LogitLensGrid } from "./LogitLensGrid";
import { LogitLensStream } from "./LogitLensStream";
import { EmptyState } from "./EmptyState";
import { LoadingOverlay } from "./LoadingOverlay";

type Mode = "grid" | "stream";

export function LogitLensSection({
  result,
  busy,
  error,
}: {
  result: LogitLensResult | null;
  busy: boolean;
  error: string | null;
}) {
  const [mode, setMode] = useState<Mode>("stream");

  if (!result) {
    return busy || error ? null : <EmptyState />;
  }

  return (
    <div className="space-y-3 relative">
      <div className="flex flex-wrap items-center gap-3">
        <p className="text-sm" style={{ fontFamily: font.ui, color: color.textMd }}>
          final prediction:{" "}
          <span style={{ fontFamily: font.mono, color: color.accent, fontWeight: 500 }}>
            {result.final_top_token.trim() || "·"}
          </span>
          {mode === "grid" && (
            <span style={{ color: color.textLo }}> · click a token column to see its journey in stream mode</span>
          )}
        </p>
        <div
          className="ml-auto flex rounded-md overflow-hidden"
          style={{ border: `1px solid ${color.border}` }}
        >
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
      </div>

      {mode === "grid" ? <LogitLensGrid result={result} /> : <LogitLensStream result={result} />}

      <LoadingOverlay active={busy} />
    </div>
  );
}
