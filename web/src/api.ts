// Talks to the FastAPI backend. Types mirror glassbox/schemas.py.

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

export async function runLogitLens(prompt: string, model: string): Promise<LogitLensResult> {
  const res = await fetch(`${BASE}/logit-lens`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ prompt, model }),
  });
  if (!res.ok) {
    let msg = await detail(res);
    if (res.status === 503) msg = `${msg} (accept the license + set HF_TOKEN)`;
    throw new ApiError(res.status, msg);
  }
  return res.json();
}
