import { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";
import type { AttentionResult } from "../api";
import { color, font, rampCss } from "../theme";
import { Select } from "./Select";
import { Explainer } from "./Explainer";

// Background intensity for a heatmap cell, weight in [0,1] — shares the ramp with the lens views.
function weightStyle(w: number): React.CSSProperties {
  return {
    backgroundColor: rampCss(w, 0.04 + 0.96 * w),
    color: w > 0.55 ? color.bg : color.textMd,
  };
}

function label(s: string): string {
  const t = s.replace(/\n/g, "⏎");
  return t.length > 6 ? t.slice(0, 6) + "…" : t;
}

// One layer/head's attention pattern, shown either as a query×key heatmap or a D3 arc diagram.
export function AttentionView({ result }: { result: AttentionResult }) {
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [mode, setMode] = useState<"heatmap" | "arcs">("arcs");

  const pattern = result.patterns[layer][head]; // [query][key]

  return (
    <div className="gb-fade-up space-y-3">
      <Explainer title="What am I looking at?">
        <p>
          Every layer reads the sentence through several <strong>attention heads</strong>. A head's
          job is to decide, for each token, which <em>earlier</em> tokens it should pull information
          from — that's how the model carries meaning across a sentence instead of reading each word
          in isolation.
        </p>
        <p>
          In <strong>arcs</strong>, each token sits on the line and an arc shows it reaching back to
          another token; brighter and thicker means a stronger pull. The <strong>heatmap</strong>{" "}
          shows the same thing as a grid — each row reaches across to the columns it attends to.
        </p>
        <p style={{ color: color.textLo }}>
          Why it's useful: this is the model's wiring diagram. Heads specialize — one tracks the
          previous word, another links a pronoun back to its noun, another copies a name it saw
          earlier. Flipping through layers and heads is exactly how researchers discovered the
          "induction heads" behind in-context learning.
        </p>
      </Explainer>

      <div className="flex flex-wrap gap-3 items-center text-sm">
        <Selector label="layer" value={layer} count={result.n_layers} onChange={setLayer} />
        <Selector label="head" value={head} count={result.n_heads} onChange={setHead} />
        <div
          className="ml-auto flex rounded-md overflow-hidden"
          style={{ border: `1px solid ${color.border}` }}
        >
          {(["arcs", "heatmap"] as const).map((m) => (
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

      <p className="text-xs" style={{ fontFamily: font.ui, color: color.textLo }}>
        Row = a token deciding where to look (query); column/target = the token it attends to (key).
        A causal model only looks backward, so the upper-right is empty.
      </p>

      {mode === "heatmap" ? (
        <Heatmap pattern={pattern} tokens={result.str_tokens} isBos={result.is_bos} />
      ) : (
        <ArcDiagram pattern={pattern} tokens={result.str_tokens} isBos={result.is_bos} />
      )}
    </div>
  );
}

function Selector({
  label,
  value,
  count,
  onChange,
}: {
  label: string;
  value: number;
  count: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex items-center gap-1" style={{ fontFamily: font.ui, color: color.textLo }}>
      {label}
      <Select
        width={56}
        mono
        ariaLabel={label}
        value={String(value)}
        onChange={(v) => onChange(Number(v))}
        options={[...Array(count).keys()].map((i) => ({ value: String(i), label: String(i) }))}
      />
    </label>
  );
}

function Heatmap({
  pattern,
  tokens,
  isBos,
}: {
  pattern: number[][];
  tokens: string[];
  isBos: boolean[];
}) {
  const n = tokens.length;
  const gridCols = `4rem repeat(${n}, minmax(2.6rem, 1fr))`;
  return (
    <div className="gb-card overflow-x-auto p-2">
      <div className="inline-grid gap-px" style={{ gridTemplateColumns: gridCols }}>
        {/* corner + key (column) header */}
        <div
          className="text-[10px] px-1 py-1 text-right"
          style={{ backgroundColor: color.bg, color: color.textDim, fontFamily: font.mono }}
        >
          q ╲ k
        </div>
        {tokens.map((t, k) => (
          <div
            key={k}
            title={t}
            className="text-[10px] px-1 py-1 text-center truncate"
            style={{
              fontFamily: font.mono,
              borderBottom: `1px solid ${color.border}`,
              color: isBos[k] ? color.textDim : color.textMd,
              fontStyle: isBos[k] ? "italic" : "normal",
            }}
          >
            {isBos[k] ? "BOS" : label(t)}
          </div>
        ))}

        {/* one row per query token */}
        {tokens.map((qt, q) => (
          <Row key={q} q={q} qt={qt} pattern={pattern} tokens={tokens} isBos={isBos} />
        ))}
      </div>
    </div>
  );
}

function Row({
  q,
  qt,
  pattern,
  tokens,
  isBos,
}: {
  q: number;
  qt: string;
  pattern: number[][];
  tokens: string[];
  isBos: boolean[];
}) {
  return (
    <>
      <div
        title={qt}
        className="sticky left-0 text-[10px] px-1 py-1 text-right truncate"
        style={{
          backgroundColor: color.bg,
          fontFamily: font.mono,
          color: isBos[q] ? color.textDim : color.textMd,
          fontStyle: isBos[q] ? "italic" : "normal",
        }}
      >
        {isBos[q] ? "BOS" : label(qt)}
      </div>
      {tokens.map((_, k) => {
        const w = pattern[q][k];
        // Causal: a query never attends to a future key — leave those blank.
        if (k > q) return <div key={k} style={{ backgroundColor: color.overlay }} />;
        return (
          <div
            key={k}
            className="text-[10px] px-1 py-1 text-center tabular-nums cursor-default"
            style={{ ...weightStyle(w), fontFamily: font.mono }}
            title={`${qt} → ${tokens[k]}\nweight: ${(w * 100).toFixed(1)}%`}
          >
            {w >= 0.005 ? (w * 100).toFixed(0) : ""}
          </div>
        );
      })}
    </>
  );
}

// D3 arc diagram: tokens on a baseline; an arc from each key up to each query, opacity = weight.
function ArcDiagram({
  pattern,
  tokens,
  isBos,
}: {
  pattern: number[][];
  tokens: string[];
  isBos: boolean[];
}) {
  const ref = useRef<SVGSVGElement | null>(null);
  const [hover, setHover] = useState<number | null>(null);

  const n = tokens.length;
  const colW = 64;
  const width = Math.max(360, n * colW);
  const height = 260;
  const baseY = height - 40;

  // Precompute visible arcs (skip negligible weights so it stays readable).
  const arcs = useMemo(() => {
    const out: { q: number; k: number; w: number }[] = [];
    for (let q = 0; q < n; q++) {
      for (let k = 0; k <= q; k++) {
        const w = pattern[q][k];
        if (w >= 0.02) out.push({ q, k, w });
      }
    }
    return out;
  }, [pattern, n]);

  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    const x = (i: number) => 24 + i * colW + colW / 2;

    // arcs
    svg
      .append("g")
      .selectAll("path")
      .data(arcs)
      .join("path")
      .attr("d", (d) => {
        const x1 = x(d.k);
        const x2 = x(d.q);
        const lift = Math.min(baseY - 16, 40 + Math.abs(d.q - d.k) * 22);
        return `M ${x1} ${baseY} Q ${(x1 + x2) / 2} ${baseY - lift} ${x2} ${baseY}`;
      })
      .attr("fill", "none")
      .attr("stroke", color.accent)
      .attr("stroke-width", (d) => 0.5 + d.w * 3)
      .attr("stroke-opacity", (d) =>
        hover == null ? 0.12 + 0.85 * d.w : d.q === hover || d.k === hover ? 0.2 + 0.8 * d.w : 0.03,
      );

    // token dots + labels on the baseline (bind indices so duplicate tokens stay distinct)
    const g = svg
      .append("g")
      .selectAll("g")
      .data(tokens.map((_, i) => i))
      .join("g")
      .attr("transform", (i) => `translate(${x(i)}, ${baseY})`)
      .style("cursor", "pointer")
      .on("mouseenter", (_, i) => setHover(i))
      .on("mouseleave", () => setHover(null));

    g.append("circle")
      .attr("r", 4)
      .attr("fill", (i) => (isBos[i] ? color.textDim : i === hover ? color.accentLight : color.accent));

    g.append("text")
      .attr("transform", "rotate(40)")
      .attr("x", 8)
      .attr("y", 4)
      .attr("font-size", 10)
      .attr("font-family", font.mono)
      .attr("fill", (i) => (isBos[i] ? color.textLo : i === hover ? color.textHi : color.textMd))
      .text((i) => (isBos[i] ? "BOS" : label(tokens[i])));
  }, [arcs, tokens, isBos, hover, colW, baseY, width]);

  return (
    <div
      className="overflow-x-auto rounded-md"
      style={{ border: `1px solid ${color.border}`, backgroundColor: color.overlay }}
    >
      <svg
        ref={ref}
        viewBox={`0 0 ${width} ${height}`}
        width={width}
        height={height}
        style={{ width: "100%", maxWidth: width, height: "auto", display: "block" }}
        preserveAspectRatio="xMinYMid meet"
      />
    </div>
  );
}
