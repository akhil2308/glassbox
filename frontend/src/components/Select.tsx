import { useEffect, useRef, useState } from "react";
import { color, font } from "../theme";

// Custom dropdown so the open list can be themed — native <select> popups
// are rendered by the OS and ignore our CSS.
export function Select<T extends string>({
  value,
  options,
  onChange,
  ariaLabel,
  mono,
  className,
  width,
}: {
  value: T;
  options: { value: T; label: string; disabled?: boolean }[];
  onChange: (v: T) => void;
  ariaLabel: string;
  mono?: boolean;
  className?: string;
  width?: number | string;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const current = options.find((o) => o.value === value);

  useEffect(() => {
    if (!open) return;
    const onDocClick = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const fontFamily = mono ? font.mono : font.ui;

  return (
    <div ref={rootRef} className={`relative inline-block ${className ?? ""}`} style={{ width }}>
      <button
        type="button"
        className="gb-select rounded-md px-2 py-1 outline-none w-full text-left flex items-center justify-between gap-2"
        style={{ fontFamily, backgroundColor: color.surface, border: `1px solid ${color.border}`, color: color.textHi }}
        aria-label={ariaLabel}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        <span>{current?.label ?? ""}</span>
        <span style={{ color: color.textLo, fontSize: 10 }}>▾</span>
      </button>
      {open && (
        <ul
          role="listbox"
          className="absolute left-0 mt-1 rounded-md py-1 z-50 overflow-y-auto max-h-60"
          style={{
            fontFamily,
            backgroundColor: color.surfaceRaised,
            border: `1px solid ${color.border}`,
            boxShadow: color.shadowCardHover,
            minWidth: "100%",
            width: "max-content",
          }}
        >
          {options.map((o) => (
            <li
              key={o.value}
              role="option"
              aria-selected={o.value === value}
              aria-disabled={o.disabled}
              className="px-2 py-1 text-sm cursor-pointer whitespace-nowrap"
              style={{
                color: o.disabled ? color.textDim : o.value === value ? color.accentLight : color.textHi,
                backgroundColor: o.value === value ? color.surfaceSunken : "transparent",
                cursor: o.disabled ? "not-allowed" : "pointer",
              }}
              onMouseDown={(e) => {
                e.preventDefault();
                if (o.disabled) return;
                onChange(o.value);
                setOpen(false);
              }}
              onMouseEnter={(e) => {
                if (!o.disabled) e.currentTarget.style.backgroundColor = color.surface;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = o.value === value ? color.surfaceSunken : "transparent";
              }}
            >
              {o.label}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
