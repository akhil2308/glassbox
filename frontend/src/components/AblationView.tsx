import { useEffect, useRef } from "react";
import * as d3 from "d3";
import type { AblationComponent, AblationResult } from "../api";
import { color, font, ramp } from "../theme";
import { Select } from "./Select";

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
    <div className="gb-fade-up space-y-3">
      <div className="flex flex-wrap gap-3 items-center text-sm">
        <label className="flex items-center gap-1" style={{ fontFamily: font.ui, color: color.textLo }}>
          ablate
          <Select
            width={144}
            mono
            ariaLabel="Component to ablate"
            value={component}
            onChange={(v) => setComponent(v as AblationComponent)}
            options={COMPONENTS}
          />
        </label>
        <span className="text-xs" style={{ fontFamily: font.ui, color: color.textLo }}>
          re-run after changing this — each bar is the model run with that one layer deleted
        </span>
      </div>

      {result && (
        <>
          <p className="text-sm" style={{ fontFamily: font.ui, color: color.textMd }}>
            baseline prediction:{" "}
            <span style={{ fontFamily: font.mono, color: color.accent, fontWeight: 500 }}>
              {result.baseline_top_token.trim() || "·"}
            </span>{" "}
            <span style={{ color: color.textLo }}>
              ({(result.baseline_top_prob * 100).toFixed(1)}%) · taller/brighter bar = deleting that
              layer hurt the prediction more
            </span>
          </p>
          <BarChart result={result} />
        </>
      )}
    </div>
  );
}

const CHART_HEIGHT = 240;

function chartWidth(nBars: number): number {
  return Math.max(360, nBars * 46);
}

function BarChart({ result }: { result: AblationResult }) {
  const ref = useRef<SVGSVGElement | null>(null);
  const width = chartWidth(result.effects.length);
  const height = CHART_HEIGHT;

  useEffect(() => {
    const base = result.baseline_top_prob;
    // "Damage" = how much probability the baseline answer lost when this layer was deleted.
    const data = result.effects.map((e) => ({
      ...e,
      drop: Math.max(0, base - e.answer_prob),
    }));

    const margin = { top: 10, right: 12, bottom: 28, left: 36 };
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
      .call((sel) => sel.selectAll("text").attr("fill", color.textMd).attr("font-size", 9).attr("font-family", font.mono))
      .call((sel) => sel.selectAll("line,path").attr("stroke", color.border));
    g.append("g")
      .call(d3.axisLeft(y).ticks(4).tickFormat((d) => `${(+d * 100).toFixed(0)}%`))
      .call((sel) => sel.selectAll("text").attr("fill", color.textMd).attr("font-size", 9).attr("font-family", font.mono))
      .call((sel) => sel.selectAll("line,path").attr("stroke", color.border));

    // bars: damage intensity via the shared confidence ramp, a gold cap marks layers
    // where the prediction flipped.
    const maxDrop = Math.max(base, d3.max(data, (d) => d.drop) ?? base) || 1;
    const damageColor = (drop: number) => {
      const [r, g2, b] = ramp(drop / maxDrop);
      return `rgb(${r},${g2},${b})`;
    };
    g.selectAll("rect.bar")
      .data(data)
      .join("rect")
      .attr("class", "bar")
      .attr("x", (d) => x(d.layer) ?? 0)
      .attr("y", (d) => y(d.drop))
      .attr("width", x.bandwidth())
      .attr("height", (d) => innerH - y(d.drop))
      .attr("fill", (d) => (d.answer_kept ? damageColor(d.drop) : color.lockGold))
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
      .attr("fill", color.textLo)
      .attr("font-size", 10)
      .attr("font-family", font.ui)
      .text("layer (gold bar = top prediction changed when this layer was deleted)");
  }, [result, width, height]);

  return (
    <div
      className="overflow-x-auto rounded-md p-2"
      style={{ border: `1px solid ${color.border}`, backgroundColor: color.overlay }}
    >
      <svg
        ref={ref}
        viewBox={`0 0 ${width} ${height}`}
        width={width}
        height={height}
        style={{ width: "100%", maxWidth: width, height: "auto", display: "block" }}
      />
    </div>
  );
}
