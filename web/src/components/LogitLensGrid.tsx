import { useState } from "react";
import type { LogitLensResult, TokenJourney } from "../api";

// Background intensity for a cell, driven by answer_prob in [0,1] (violet on dark).
function cellStyle(p: number): React.CSSProperties {
  return {
    backgroundColor: `rgba(139, 92, 246, ${0.08 + 0.92 * p})`,
    color: p > 0.55 ? "#0b0f17" : "#cbd5e1",
  };
}

function trunc(s: string): string {
  const t = s.replace(/\n/g, "⏎");
  return t.length > 7 ? t.slice(0, 7) + "…" : t;
}

// Rows = layers (top = full model output, bottom = embeddings). Columns = prompt tokens.
// Cell shows the layer's top-predicted token; color = answer_prob. BOS column is dimmed/labeled.
export function LogitLensGrid({ result }: { result: LogitLensResult }) {
  const [pinned, setPinned] = useState<number | null>(null);
  const tokens = result.tokens;
  const nLayers = tokens[0].layers.length;
  // Top row = highest layer, so the answer "climbs upward" as you read down-to-up.
  const layerOrder = [...Array(nLayers).keys()].reverse();

  const gridCols = `3.5rem repeat(${tokens.length}, minmax(3.2rem, 1fr))`;

  return (
    <div className="flex gap-6 items-start">
      <div className="overflow-x-auto">
        <div className="inline-grid gap-px" style={{ gridTemplateColumns: gridCols }}>
          {/* header: token strings */}
          <div className="sticky left-0 bg-slate-900 text-[10px] text-slate-500 px-1 py-1 text-right">
            layer
          </div>
          {tokens.map((t) => (
            <button
              key={t.position}
              onClick={() => setPinned(t.is_bos ? null : t.position)}
              title={t.is_bos ? "beginning-of-sequence (no meaningful journey)" : t.str_token}
              className={
                "text-[10px] px-1 py-1 truncate text-center border-b border-slate-700 " +
                (t.is_bos
                  ? "text-slate-600 italic"
                  : pinned === t.position
                  ? "text-violet-300 font-semibold"
                  : "text-slate-300 hover:text-violet-300")
              }
            >
              {t.is_bos ? "BOS" : trunc(t.str_token)}
            </button>
          ))}

          {/* body: one row per layer (top = full model) */}
          {layerOrder.map((li) => (
            <Row key={li} layerIndex={li} tokens={tokens} />
          ))}
        </div>
      </div>

      {pinned != null && <JourneyPanel token={tokens[pinned]} onClose={() => setPinned(null)} />}
    </div>
  );
}

function Row({ layerIndex, tokens }: { layerIndex: number; tokens: TokenJourney[] }) {
  const label = tokens[0].layers[layerIndex].label;
  return (
    <>
      <div
        className="sticky left-0 bg-slate-900 text-[10px] text-slate-500 px-1 py-1 text-right tabular-nums"
        title={label}
      >
        {layerIndex}
      </div>
      {tokens.map((t) => {
        const lp = t.layers[layerIndex];
        if (t.is_bos) {
          return <div key={t.position} className="bg-slate-800/40" />;
        }
        return (
          <div
            key={t.position}
            className="text-[10px] px-1 py-1 text-center truncate cursor-default"
            style={cellStyle(lp.answer_prob)}
            title={`token: ${lp.top_token}\np(top): ${(lp.top_prob * 100).toFixed(1)}%\np(answer): ${(lp.answer_prob * 100).toFixed(1)}%`}
          >
            {trunc(lp.top_token)}
          </div>
        );
      })}
    </>
  );
}

function JourneyPanel({ token, onClose }: { token: TokenJourney; onClose: () => void }) {
  return (
    <div className="w-64 shrink-0 rounded-md border border-slate-700 bg-slate-900 p-3 text-xs">
      <div className="flex justify-between items-center mb-2">
        <span className="text-slate-300">
          journey of <span className="text-violet-300">{token.str_token.trim() || "·"}</span>
        </span>
        <button onClick={onClose} className="text-slate-500 hover:text-slate-200">
          ✕
        </button>
      </div>
      <div className="space-y-px">
        {[...token.layers].reverse().map((lp) => (
          <div key={lp.layer} className="flex items-center gap-2">
            <span className="w-5 text-right text-slate-500 tabular-nums">{lp.layer}</span>
            <span className="flex-1 truncate text-slate-300" title={lp.top_token}>
              {lp.top_token.trim() || "·"}
            </span>
            <span className="tabular-nums text-slate-400">
              {(lp.answer_prob * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
