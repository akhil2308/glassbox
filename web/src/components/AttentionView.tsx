import { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";
import type { AttentionResult } from "../api";

// Background intensity for a heatmap cell, weight in [0,1] (violet on dark) — matches the lens grid.
function weightStyle(w: number): React.CSSProperties {
  return {
    backgroundColor: `rgba(139, 92, 246, ${0.04 + 0.96 * w})`,
    color: w > 0.55 ? "#0b0f17" : "#cbd5e1",
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
    <div className="space-y-3">
      <div className="flex flex-wrap gap-3 items-center text-sm">
        <Selector label="layer" value={layer} count={result.n_layers} onChange={setLayer} />
        <Selector label="head" value={head} count={result.n_heads} onChange={setHead} />
        <div className="ml-auto flex rounded-md border border-slate-700 overflow-hidden">
          {(["arcs", "heatmap"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={
                "px-3 py-1 text-xs " +
                (mode === m ? "bg-violet-600 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700")
              }
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <p className="text-xs text-slate-500">
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
    <label className="flex items-center gap-1 text-slate-400">
      {label}
      <select
        className="rounded-md bg-slate-800 border border-slate-700 px-2 py-1 outline-none text-slate-200"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      >
        {[...Array(count).keys()].map((i) => (
          <option key={i} value={i}>
            {i}
          </option>
        ))}
      </select>
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
    <div className="overflow-x-auto">
      <div className="inline-grid gap-px" style={{ gridTemplateColumns: gridCols }}>
        {/* corner + key (column) header */}
        <div className="bg-slate-900 text-[10px] text-slate-600 px-1 py-1 text-right">q ╲ k</div>
        {tokens.map((t, k) => (
          <div
            key={k}
            title={t}
            className={
              "text-[10px] px-1 py-1 text-center truncate border-b border-slate-700 " +
              (isBos[k] ? "text-slate-600 italic" : "text-slate-300")
            }
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
        className={
          "sticky left-0 bg-slate-900 text-[10px] px-1 py-1 text-right truncate " +
          (isBos[q] ? "text-slate-600 italic" : "text-slate-400")
        }
      >
        {isBos[q] ? "BOS" : label(qt)}
      </div>
      {tokens.map((_, k) => {
        const w = pattern[q][k];
        // Causal: a query never attends to a future key — leave those blank.
        if (k > q) return <div key={k} className="bg-slate-900/40" />;
        return (
          <div
            key={k}
            className="text-[10px] px-1 py-1 text-center tabular-nums cursor-default"
            style={weightStyle(w)}
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
      .attr("stroke", "#8b5cf6")
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
      .attr("fill", (i) => (isBos[i] ? "#475569" : i === hover ? "#c4b5fd" : "#8b5cf6"));

    g.append("text")
      .attr("transform", "rotate(40)")
      .attr("x", 8)
      .attr("y", 4)
      .attr("font-size", 10)
      .attr("fill", (i) => (isBos[i] ? "#64748b" : i === hover ? "#ddd6fe" : "#cbd5e1"))
      .text((i) => (isBos[i] ? "BOS" : label(tokens[i])));
  }, [arcs, tokens, isBos, hover, colW, baseY, width]);

  return (
    <div className="overflow-x-auto rounded-md border border-slate-800 bg-slate-900/40">
      <svg ref={ref} width={width} height={height} />
    </div>
  );
}
