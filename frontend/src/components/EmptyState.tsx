import { color, font } from "../theme";

export function EmptyState({ message = "Enter a prompt and hit Run." }: { message?: string }) {
  return (
    <p
      className="text-sm text-center py-6"
      style={{
        color: color.textLo,
        fontFamily: font.mono,
        textDecoration: "underline",
        textDecorationColor: color.accent,
        textDecorationThickness: "1px",
        textUnderlineOffset: "6px",
        opacity: 0.85,
      }}
    >
      {message}
    </p>
  );
}
