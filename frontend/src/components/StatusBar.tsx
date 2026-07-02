import type { ModelArch } from "../api";
import { color, font } from "../theme";

// Compact, e.g. "RMSNorm · GQA 4:1 · 262k vocab · soft-cap". Vocab is the headline difference.
function archSummary(a: ModelArch): string {
  const vocab = a.vocab >= 1000 ? `${Math.round(a.vocab / 1000)}k vocab` : `${a.vocab} vocab`;
  return [a.norm, a.attention, vocab, a.soft_cap ? "soft-cap" : null].filter(Boolean).join(" · ");
}

// Active model, device, and last request latency.
export function StatusBar({
  model,
  device,
  latencyMs,
  arch,
}: {
  model: string | null;
  device: string | null;
  latencyMs: number | null;
  arch: ModelArch | null;
}) {
  const chips: { label: string; value: string }[] = [
    { label: "model", value: model ?? "—" },
    ...(arch ? [{ label: "arch", value: archSummary(arch) }] : []),
    { label: "device", value: device ?? "—" },
    { label: "latency", value: latencyMs == null ? "—" : `${latencyMs} ms` },
  ];

  return (
    <div className="flex gap-2 text-xs">
      {chips.map((c) => (
        <span
          key={c.label}
          className="inline-flex items-center gap-1.5 rounded-md"
          style={{ backgroundColor: color.surfaceRaised, padding: "2px 8px" }}
        >
          <span style={{ fontFamily: font.ui, color: color.textMd }}>{c.label}</span>
          <span style={{ fontFamily: font.mono, color: color.textHi }}>{c.value}</span>
        </span>
      ))}
    </div>
  );
}
