import type { CSSProperties } from "react";

// Single source of truth for color/typography. Keep frontend/src/index.css's
// :root block in sync with `color`/`font` by hand — CSS vars there exist only
// for plain-CSS hover/focus rules that can't easily reach inline styles.
export const color = {
  bg: "#0d1b1e",
  surface: "#13282c",
  surfaceRaised: "#172e33",
  surfaceSunken: "rgba(19, 40, 44, 0.4)",
  border: "#1f3a3f",
  textHi: "#eafff6",
  textMd: "#c4d6d4",
  textLo: "#5b7a7e",
  textDim: "#3a5358",
  accent: "#34d3bc",
  accentLight: "#7ee8d8",
  danger: "#f2545b",
  dangerBg: "rgba(89, 30, 33, 0.5)",
  overlay: "rgba(13, 27, 30, 0.4)",
  overlayStrong: "rgba(13, 27, 30, 0.62)",
  lockGold: "rgb(255, 207, 92)",
  lockGoldDim: "rgba(255, 207, 92, 0.5)",
  shadowCard: "0 1px 2px rgba(0,0,0,0.3), 0 8px 24px -8px rgba(0,0,0,0.45)",
  shadowCardHover: "0 2px 4px rgba(0,0,0,0.35), 0 12px 32px -8px rgba(0,0,0,0.55)",
} as const;

export const font = {
  ui: "'Space Grotesk', ui-sans-serif, system-ui, sans-serif",
  mono: "'JetBrains Mono', ui-monospace, 'SF Mono', monospace",
} as const;

// Confidence ramp: cold teal (noise) -> bright teal (forming/bright/near) -> gold (locked).
const RAMP_STOPS: [number, [number, number, number]][] = [
  [0.0, [31, 74, 79]],
  [0.3, [38, 120, 110]],
  [0.55, [52, 211, 176]],
  [0.8, [150, 226, 150]],
  [1.0, [255, 207, 92]],
];

export function ramp(p: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, p));
  for (let i = 1; i < RAMP_STOPS.length; i++) {
    const [p0, c0] = RAMP_STOPS[i - 1];
    const [p1, c1] = RAMP_STOPS[i];
    if (x <= p1) {
      const t = (x - p0) / (p1 - p0 || 1);
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * t),
        Math.round(c0[1] + (c1[1] - c0[1]) * t),
        Math.round(c0[2] + (c1[2] - c0[2]) * t),
      ];
    }
  }
  return RAMP_STOPS[RAMP_STOPS.length - 1][1];
}

export function rampCss(p: number, alpha = 1): string {
  const [r, g, b] = ramp(p);
  return `rgba(${r},${g},${b},${alpha})`;
}

// Shared cell background for probability-driven grids (lens grid, attention heatmap).
export function probCellStyle(p: number): CSSProperties {
  return {
    backgroundColor: rampCss(p, 0.16 + 0.8 * Math.max(0, Math.min(1, p))),
    color: p > 0.55 ? color.bg : color.textMd,
  };
}

// Glow strength scales with confidence — shared by the lens stream's prediction
// boxes and any other element that should "light up" as probability rises.
export function glow(p: number): string {
  const x = Math.max(0, Math.min(1, p));
  return `0 0 ${6 + x * 26}px ${rampCss(x, 0.25 + 0.5 * x)}`;
}
