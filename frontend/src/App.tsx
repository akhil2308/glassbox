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
import { LogitLensSection } from "./components/LogitLensSection";
import { AttentionView } from "./components/AttentionView";
import { AblationView } from "./components/AblationView";
import { EmptyState } from "./components/EmptyState";
import { LoadingOverlay } from "./components/LoadingOverlay";
import { color, font } from "./theme";

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
      <header className="flex items-baseline gap-2">
        <span style={{ color: color.accent, fontSize: "1.5rem", lineHeight: 1 }}>◆</span>
        <div>
          <h1
            className="text-[1.75rem] font-bold"
            style={{ fontFamily: font.ui, color: color.textHi, letterSpacing: "0.02em" }}
          >
            GLASSBOX
          </h1>
          <p className="text-sm" style={{ fontFamily: font.ui, color: color.textMd }}>
            {blurb}
          </p>
        </div>
      </header>

      <nav
        role="tablist"
        className="inline-flex gap-1 rounded-lg p-1"
        style={{ backgroundColor: color.surfaceRaised }}
      >
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={view === t.id}
            aria-controls={`panel-${t.id}`}
            onClick={() => setView(t.id)}
            className="gb-nav-tab gb-btn rounded-md px-3 py-1.5 text-sm"
            style={{
              fontFamily: font.ui,
              backgroundColor: view === t.id ? color.surface : "transparent",
              color: view === t.id ? color.accent : color.textMd,
              boxShadow: view === t.id ? color.shadowCard : "none",
            }}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <div className="gb-card p-4 space-y-3">
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
      </div>

      {error && (
        <div
          role="alert"
          className="gb-fade-up rounded-md text-sm px-3 py-2"
          style={{
            fontFamily: font.ui,
            border: `1px solid ${color.danger}`,
            backgroundColor: color.dangerBg,
            color: color.danger,
          }}
        >
          <span aria-hidden="true">⚠ </span>
          {error}
        </div>
      )}

      <div role="tabpanel" id={`panel-${view}`}>
        {view === "lens" && <LogitLensSection result={lens} busy={busy} error={error} />}

        {view === "attention" &&
          (attn ? (
            <div className="relative">
              <AttentionView result={attn} />
              <LoadingOverlay active={busy} />
            </div>
          ) : busy || error ? null : (
            <EmptyState />
          ))}

        {view === "ablation" && (
          <div className="relative">
            <AblationView result={abl} component={component} setComponent={setComponent} />
            <LoadingOverlay active={busy && abl != null} />
          </div>
        )}
      </div>
    </div>
  );
}
