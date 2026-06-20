import { color, font } from "../theme";

// Active model, device, and last request latency.
export function StatusBar({
  model,
  device,
  latencyMs,
}: {
  model: string | null;
  device: string | null;
  latencyMs: number | null;
}) {
  const chips: { label: string; value: string }[] = [
    { label: "model", value: model ?? "—" },
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
