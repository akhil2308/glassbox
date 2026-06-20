import type { LogitLensResult, TokenJourney } from "../api";
import { color, font, probCellStyle } from "../theme";

function trunc(s: string): string {
  const t = s.replace(/\n/g, "⏎");
  return t.length > 7 ? t.slice(0, 7) + "…" : t;
}

// Rows = layers (top = full model output, bottom = embeddings). Columns = prompt tokens.
// Cell shows the layer's top-predicted token; color = answer_prob. BOS column is dimmed/labeled.
export function LogitLensGrid({ result }: { result: LogitLensResult }) {
  const tokens = result.tokens;
  const nLayers = tokens[0].layers.length;
  // Top row = highest layer, so the answer "climbs upward" as you read down-to-up.
  const layerOrder = [...Array(nLayers).keys()].reverse();

  const gridCols = `3.5rem repeat(${tokens.length}, minmax(3.2rem, 1fr))`;

  return (
    <div className="overflow-x-auto">
      <div className="inline-grid gap-px" style={{ gridTemplateColumns: gridCols }}>
        {/* header: token strings */}
        <div
          className="sticky left-0 text-[10px] px-1 py-1 text-right"
          style={{ backgroundColor: color.bg, color: color.textLo, fontFamily: font.mono }}
        >
          layer
        </div>
        {tokens.map((t) => (
          <div
            key={t.position}
            title={t.is_bos ? "beginning-of-sequence (no meaningful journey)" : t.str_token}
            className="text-[10px] px-1 py-1 truncate text-center"
            style={{
              fontFamily: font.mono,
              borderBottom: `1px solid ${color.border}`,
              color: t.is_bos ? color.textDim : color.textMd,
              fontStyle: t.is_bos ? "italic" : "normal",
            }}
          >
            {t.is_bos ? "BOS" : trunc(t.str_token)}
          </div>
        ))}

        {/* body: one row per layer (top = full model) */}
        {layerOrder.map((li) => (
          <Row key={li} layerIndex={li} tokens={tokens} />
        ))}
      </div>
    </div>
  );
}

function Row({ layerIndex, tokens }: { layerIndex: number; tokens: TokenJourney[] }) {
  const label = tokens[0].layers[layerIndex].label;
  return (
    <>
      <div
        className="sticky left-0 text-[10px] px-1 py-1 text-right tabular-nums"
        style={{ backgroundColor: color.bg, color: color.textLo, fontFamily: font.mono }}
        title={label}
      >
        {layerIndex}
      </div>
      {tokens.map((t) => {
        const lp = t.layers[layerIndex];
        if (t.is_bos) {
          return <div key={t.position} style={{ backgroundColor: color.surfaceSunken }} />;
        }
        return (
          <div
            key={t.position}
            className="gb-card-raised text-[10px] px-1 py-1 text-center truncate cursor-default"
            style={{ ...probCellStyle(lp.answer_prob), fontFamily: font.mono, border: "1px solid transparent" }}
            title={`token: ${lp.top_token}\np(top): ${(lp.top_prob * 100).toFixed(1)}%\np(answer): ${(lp.answer_prob * 100).toFixed(1)}%`}
          >
            {trunc(lp.top_token)}
          </div>
        );
      })}
    </>
  );
}
