import { useEffect, useState } from "react";
import {
  ApiError,
  fetchHealth,
  fetchModels,
  runLogitLens,
  type LogitLensResult,
  type ModelInfo,
} from "./api";
import { PromptBar } from "./components/PromptBar";
import { StatusBar } from "./components/StatusBar";
import { LogitLensGrid } from "./components/LogitLensGrid";

export default function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState("gpt2");
  const [prompt, setPrompt] = useState("The capital of France is");
  const [result, setResult] = useState<LogitLensResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [device, setDevice] = useState<string | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);

  useEffect(() => {
    fetchModels().then(setModels).catch((e) => setError(String(e.message ?? e)));
    fetchHealth().then((h) => setDevice(h.device)).catch(() => {});
  }, []);

  async function onRun() {
    setBusy(true);
    setError(null);
    const t0 = performance.now();
    try {
      const res = await runLogitLens(prompt, model);
      setResult(res);
      setLatencyMs(Math.round(performance.now() - t0));
      // a freshly-loaded model flips loaded -> true; refresh the list
      fetchModels().then(setModels).catch(() => {});
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : String((e as Error).message ?? e);
      setError(msg);
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-4">
      <header>
        <h1 className="text-2xl font-semibold text-slate-100">GLASSBOX</h1>
        <p className="text-sm text-slate-400">
          The logit lens — watch a prediction assemble layer by layer up the residual stream.
        </p>
      </header>

      <PromptBar
        prompt={prompt}
        setPrompt={setPrompt}
        models={models}
        model={model}
        setModel={setModel}
        onRun={onRun}
        busy={busy}
      />

      <StatusBar model={result?.model_name ?? null} device={device} latencyMs={latencyMs} />

      {error && (
        <div className="rounded-md border border-red-800 bg-red-950/50 text-red-300 text-sm px-3 py-2">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-2">
          <p className="text-sm text-slate-400">
            final prediction:{" "}
            <span className="text-violet-300 font-medium">
              {result.final_top_token.trim() || "·"}
            </span>
            <span className="text-slate-600"> · click a token column to pin its journey</span>
          </p>
          <LogitLensGrid result={result} />
        </div>
      )}

      {!result && !error && !busy && (
        <p className="text-sm text-slate-500">Enter a prompt and hit Run.</p>
      )}
    </div>
  );
}
