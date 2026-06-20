import type { ReactNode } from "react";
import { color, font } from "../theme";

// Collapsible "what am I looking at?" panel. Native <details> — no JS, accessible,
// remembers nothing on purpose (open by default so first-timers read it).
export function Explainer({ title, children }: { title: string; children: ReactNode }) {
  return (
    <details
      open
      className="gb-card rounded-md px-3 py-2"
      style={{ backgroundColor: color.surfaceSunken, border: `1px solid ${color.border}` }}
    >
      <summary
        className="cursor-pointer text-sm font-semibold select-none"
        style={{ fontFamily: font.ui, color: color.accent }}
      >
        {title}
      </summary>
      <div
        className="mt-2 space-y-2 text-sm leading-relaxed"
        style={{ fontFamily: font.ui, color: color.textMd }}
      >
        {children}
      </div>
    </details>
  );
}
