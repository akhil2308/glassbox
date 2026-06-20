import { color, font } from "../theme";

export function EmptyState({ message = "Enter a prompt and hit Run." }: { message?: string }) {
  return (
    <div
      className="rounded-md py-10 text-center"
      style={{ border: `1px dashed ${color.border}` }}
    >
      <p
        className="text-sm"
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
    </div>
  );
}
