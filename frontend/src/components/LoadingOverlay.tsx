import { color, font } from "../theme";

export function LoadingOverlay({ active }: { active: boolean }) {
  if (!active) return null;
  return (
    <div
      className="absolute inset-0 flex items-center justify-center z-10"
      style={{ backgroundColor: "rgba(13,27,30,0.55)" }}
    >
      <span style={{ fontFamily: font.mono, color: color.textHi, fontSize: 13 }}>running ↑</span>
    </div>
  );
}
