// Talks to the FastAPI backend. Types mirror backend/app/schemas/results.py.

const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface ModelInfo {
  name: string;
  display_name: string;
  gated: boolean;
  loaded: boolean;
}

export interface LayerPrediction {
  layer: number;
  label: string;
  top_token: string;
  top_prob: number;
  answer_prob: number;
}

export interface TokenJourney {
  position: number;
  str_token: string;
  is_bos: boolean;
  layers: LayerPrediction[];
}

export interface LogitLensResult {
  prompt: string;
  model_name: string;
  str_tokens: string[];
  final_top_token: string;
  tokens: TokenJourney[];
}

export interface AttentionResult {
  model_name: string;
  prompt: string;
  str_tokens: string[];
  is_bos: boolean[];
  n_layers: number;
  n_heads: number;
  // patterns[layer][head][query][key]
  patterns: number[][][][];
}

export interface AblationEffect {
  layer: number;
  ablated_top_token: string;
  ablated_top_prob: number;
  answer_prob: number;
  answer_kept: boolean;
}

export interface AblationResult {
  model_name: string;
  prompt: string;
  str_tokens: string[];
  component: string;
  baseline_top_token: string;
  baseline_top_prob: number;
  effects: AblationEffect[];
}

export type AblationComponent = "block" | "attn" | "mlp";

// Thrown with a user-facing message derived from the API's status + detail.
export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function detail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d: { msg: string }) => d.msg).join("; ");
  } catch {
    /* fall through */
  }
  return res.statusText;
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${BASE}/models`);
  if (!res.ok) throw new ApiError(res.status, await detail(res));
  return res.json();
}

export async function fetchHealth(): Promise<{ device: string; loaded_models: string[] }> {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new ApiError(res.status, await detail(res));
  return res.json();
}

// Shared POST: serialize body, raise a friendly ApiError on failure (gated models get a hint).
async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let msg = await detail(res);
    if (res.status === 503) msg = `${msg} (accept the license + set HF_TOKEN)`;
    throw new ApiError(res.status, msg);
  }
  return res.json();
}

export function runLogitLens(prompt: string, model: string): Promise<LogitLensResult> {
  return post("/logit-lens", { prompt, model });
}

export function runAttention(prompt: string, model: string): Promise<AttentionResult> {
  return post("/attention", { prompt, model });
}

export function runAblation(
  prompt: string,
  model: string,
  component: AblationComponent,
): Promise<AblationResult> {
  return post("/ablation", { prompt, model, component });
}
