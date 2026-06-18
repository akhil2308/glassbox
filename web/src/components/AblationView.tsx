import { useEffect, useRef } from "react";
import * as d3 from "d3";
import type { AblationComponent, AblationResult } from "../api";

const COMPONENTS: { value: AblationComponent; label: string }[] = [
  { value: "block", label: "whole block" },
  { value: "attn", label: "attention only" },
  { value: "mlp", label: "MLP only" },
];

// Per-layer ablation effect: how much deleting each layer hurts the model's own prediction.
export function AblationView({
  result,
  component,
  setComponent,
}: {
  result: AblationResult | null;
  component: AblationComponent;
  setComponent: (c: AblationComponent) => void;
}) {
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-3 items-center text-sm">
        <label className="flex items-center gap-1 text-slate-400">
          ablate
          <select
            className="rounded-md bg-slate-800 border border-slate-700 px-2 py-1 outline-none text-slate-200"
            value={component}
            onChange={(e) => setComponent(e.target.value as AblationComponent)}
          >
            {COMPONENTS.map((c) => (
              <option key={c.value} value={c.value}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
        <span className="text-xs text-slate-500">
          re-run after changing this — each bar is the model run with that one layer deleted
        </span>
      </div>

      {result && (
        <>
          <p className="text-sm text-slate-400">
            baseline prediction:{" "}
            <span className="text-violet-300 font-medium">
              {result.baseline_top_token.trim() || "·"}
            </span>{" "}
            <span className="text-slate-600">
              ({(result.baseline_top_prob * 100).toFixed(1)}%) · taller/redder bar = deleting that
              layer hurt the prediction more
            </span>
          </p>
          <BarChart result={result} />
        </>
      )}
    </div>
  );
}

function BarChart({ result }: { result: AblationResult }) {
  const ref = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    const base = result.baseline_top_prob;
    // "Damage" = how much probability the baseline answer lost when this layer was deleted.
    const data = result.effects.map((e) => ({
      ...e,
      drop: Math.max(0, base - e.answer_prob),
    }));

    const margin = { top: 10, right: 12, bottom: 28, left: 36 };
    const width = Math.max(360, data.length * 46);
    const height = 240;
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3
      .scaleBand<number>()
      .domain(data.map((d) => d.layer))
      .range([0, innerW])
      .padding(0.2);
    const y = d3
      .scaleLinear()
      .domain([0, Math.max(base, d3.max(data, (d) => d.drop) ?? base) || 1])
      .range([innerH, 0]);

    // axes
    g.append("g")
      .attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(x).tickSizeOuter(0))
      .call((sel) => sel.selectAll("text").attr("fill", "#94a3b8").attr("font-size", 9))
      .call((sel) => sel.selectAll("line,path").attr("stroke", "#334155"));
    g.append("g")
      .call(d3.axisLeft(y).ticks(4).tickFormat((d) => `${(+d * 100).toFixed(0)}%`))
      .call((sel) => sel.selectAll("text").attr("fill", "#94a3b8").attr("font-size", 9))
      .call((sel) => sel.selectAll("line,path").attr("stroke", "#334155"));

    // bars: red intensity tracks damage; a gold cap marks layers where the prediction flipped.
    const color = d3.scaleSequential(d3.interpolateInferno).domain([0, base || 1]);
    g.selectAll("rect.bar")
      .data(data)
      .join("rect")
      .attr("class", "bar")
      .attr("x", (d) => x(d.layer) ?? 0)
      .attr("y", (d) => y(d.drop))
      .attr("width", x.bandwidth())
      .attr("height", (d) => innerH - y(d.drop))
      .attr("fill", (d) => (d.answer_kept ? color(d.drop) : "#f59e0b"))
      .append("title")
      .text(
        (d) =>
          `layer ${d.layer}\nablated top: ${d.ablated_top_token} (${(d.ablated_top_prob * 100).toFixed(1)}%)\n` +
          `answer prob: ${(d.answer_prob * 100).toFixed(1)}%\n${d.answer_kept ? "prediction held" : "prediction BROKE"}`,
      );

    g.append("text")
      .attr("x", innerW / 2)
      .attr("y", innerH + 26)
      .attr("text-anchor", "middle")
      .attr("fill", "#64748b")
      .attr("font-size", 10)
      .text("layer (gold bar = top prediction changed when this layer was deleted)");
  }, [result]);

  return (
    <div className="overflow-x-auto rounded-md border border-slate-800 bg-slate-900/40 p-2">
      <svg ref={ref} width={Math.max(360, result.effects.length * 46)} height={240} />
    </div>
  );
}
