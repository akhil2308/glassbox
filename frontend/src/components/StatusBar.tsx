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
  return (
    <div className="flex gap-4 text-xs" style={{ fontFamily: font.ui, color: color.textLo }}>
      <span>
        model: <span style={{ fontFamily: font.mono, color: color.textHi }}>{model ?? "—"}</span>
      </span>
      <span>
        device: <span style={{ fontFamily: font.mono, color: color.textHi }}>{device ?? "—"}</span>
      </span>
      <span>
        latency:{" "}
        <span style={{ fontFamily: font.mono, color: color.textHi }}>
          {latencyMs == null ? "—" : `${latencyMs} ms`}
        </span>
      </span>
    </div>
  );
}
