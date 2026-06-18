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
    <div className="flex gap-4 text-xs text-slate-400">
      <span>
        model: <span className="text-slate-200">{model ?? "—"}</span>
      </span>
      <span>
        device: <span className="text-slate-200">{device ?? "—"}</span>
      </span>
      <span>
        latency:{" "}
        <span className="text-slate-200">
          {latencyMs == null ? "—" : `${latencyMs} ms`}
        </span>
      </span>
    </div>
  );
}
