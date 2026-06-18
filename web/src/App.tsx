import { useEffect, useState } from "react";
import {
  ApiError,
  fetchHealth,
  fetchModels,
  runAblation,
  runAttention,
  runLogitLens,
  type AblationComponent,
  type AblationResult,
  type AttentionResult,
  type LogitLensResult,
  type ModelInfo,
} from "./api";
import { PromptBar } from "./components/PromptBar";
import { StatusBar } from "./components/StatusBar";
import { LogitLensGrid } from "./components/LogitLensGrid";
import { AttentionView } from "./components/AttentionView";
import { AblationView } from "./components/AblationView";

type View = "lens" | "attention" | "ablation";

const TABS: { id: View; label: string; blurb: string }[] = [
  {
    id: "lens",
    label: "Logit lens",
    blurb: "Watch a prediction assemble layer by layer up the residual stream.",
  },
  {
    id: "attention",
    label: "Attention",
    blurb: "See which tokens look at which, per layer and head.",
  },
  {
    id: "ablation",
    label: "Ablation",
    blurb: "Delete a layer and watch how much the prediction breaks.",
  },
];

export default function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState("gpt2");
  const [prompt, setPrompt] = useState("The capital of France is");
  const [view, setView] = useState<View>("lens");

  // One result slot per view — switching tabs keeps each view's last render.
  const [lens, setLens] = useState<LogitLensResult | null>(null);
  const [attn, setAttn] = useState<AttentionResult | null>(null);
  const [abl, setAbl] = useState<AblationResult | null>(null);
  const [component, setComponent] = useState<AblationComponent>("block");

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
      if (view === "lens") setLens(await runLogitLens(prompt, model));
      else if (view === "attention") setAttn(await runAttention(prompt, model));
      else setAbl(await runAblation(prompt, model, component));
      setLatencyMs(Math.round(performance.now() - t0));
      // a freshly-loaded model flips loaded -> true; refresh the list
      fetchModels().then(setModels).catch(() => {});
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : String((e as Error).message ?? e);
      setError(msg);
    } finally {
      setBusy(false);
    }
  }

  const activeModelName =
    view === "lens" ? lens?.model_name : view === "attention" ? attn?.model_name : abl?.model_name;
  const blurb = TABS.find((t) => t.id === view)!.blurb;

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-4">
      <header>
        <h1 className="text-2xl font-semibold text-slate-100">GLASSBOX</h1>
        <p className="text-sm text-slate-400">{blurb}</p>
      </header>

      <nav className="flex gap-1 border-b border-slate-800">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setView(t.id)}
            className={
              "px-3 py-2 text-sm -mb-px border-b-2 " +
              (view === t.id
                ? "border-violet-500 text-violet-300"
                : "border-transparent text-slate-400 hover:text-slate-200")
            }
          >
            {t.label}
          </button>
        ))}
      </nav>

      <PromptBar
        prompt={prompt}
        setPrompt={setPrompt}
        models={models}
        model={model}
        setModel={setModel}
        onRun={onRun}
        busy={busy}
      />

      <StatusBar model={activeModelName ?? null} device={device} latencyMs={latencyMs} />

      {error && (
        <div className="rounded-md border border-red-800 bg-red-950/50 text-red-300 text-sm px-3 py-2">
          {error}
        </div>
      )}

      {view === "lens" &&
        (lens ? (
          <div className="space-y-2">
            <p className="text-sm text-slate-400">
              final prediction:{" "}
              <span className="text-violet-300 font-medium">
                {lens.final_top_token.trim() || "·"}
              </span>
              <span className="text-slate-600"> · click a token column to pin its journey</span>
            </p>
            <LogitLensGrid result={lens} />
          </div>
        ) : (
          <Empty busy={busy} error={error} />
        ))}

      {view === "attention" &&
        (attn ? <AttentionView result={attn} /> : <Empty busy={busy} error={error} />)}

      {view === "ablation" && (
        <AblationView result={abl} component={component} setComponent={setComponent} />
      )}
    </div>
  );
}

function Empty({ busy, error }: { busy: boolean; error: string | null }) {
  if (busy || error) return null;
  return <p className="text-sm text-slate-500">Enter a prompt and hit Run.</p>;
}
