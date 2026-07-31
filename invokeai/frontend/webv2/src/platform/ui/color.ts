/**
 * Hex-string ↔ object color conversion for the workbench.
 *
 * The canonical object form is {@link RgbaColor}: `r`/`g`/`b` are integers in
 * `[0, 255]` and `a` is a unit float in `[0, 1]`. That matches the generation
 * settings' `infillColorValue` and the legacy app's `zRgbaColor`, so those
 * consumers adapt with no extra arithmetic. The one consumer that differs —
 * workflow `ColorField`, whose alpha is `[0, 255]` — scales at its own call
 * site rather than forcing a second alpha convention through here.
 *
 * Parsing is deliberately tolerant: these values come from persisted documents,
 * node fields, and hand-typed input, so anything unparseable falls back rather
 * than throwing. Formatting always emits lowercase with a leading `#`.
 *
 * Pure module — no React, no Chakra, no DOM.
 */

/** An RGBA color. `r`/`g`/`b` are `[0, 255]` integers; `a` is a `[0, 1]` float. */
export interface RgbaColor {
  r: number;
  g: number;
  b: number;
  a: number;
}

export interface FormatHexColorOptions {
  /** Emit `#rrggbbaa` instead of `#rrggbb`. */
  alpha?: boolean;
}

/** Opaque black — the fallback for anything unparseable. */
export const BLACK: RgbaColor = { a: 1, b: 0, g: 0, r: 0 };

/** `#rgb`, `#rgba`, `#rrggbb`, or `#rrggbbaa`, with an optional leading `#`. */
const HEX_PATTERN = /^#?([0-9a-f]{3}|[0-9a-f]{4}|[0-9a-f]{6}|[0-9a-f]{8})$/i;

const clampChannel = (value: number): number => {
  if (!Number.isFinite(value)) {
    return 0;
  }

  return Math.min(255, Math.max(0, Math.round(value)));
};

const clampAlpha = (value: number): number => {
  if (!Number.isFinite(value)) {
    return 1;
  }

  return Math.min(1, Math.max(0, value));
};

const toHexByte = (value: number): string => clampChannel(value).toString(16).padStart(2, '0');

/** Expands a shorthand nibble (`f`) into a byte (`ff`). */
const expandNibble = (nibble: string): number => parseInt(`${nibble}${nibble}`, 16);

/** Parses any supported hex form, or `null` when the input is not one. */
const tryParseHexColor = (value: string): RgbaColor | null => {
  const match = HEX_PATTERN.exec(value.trim());

  if (!match) {
    return null;
  }

  const digits = match[1];

  if (digits.length <= 4) {
    return {
      a: digits.length === 4 ? expandNibble(digits[3]) / 255 : 1,
      b: expandNibble(digits[2]),
      g: expandNibble(digits[1]),
      r: expandNibble(digits[0]),
    };
  }

  return {
    a: digits.length === 8 ? parseInt(digits.slice(6, 8), 16) / 255 : 1,
    b: parseInt(digits.slice(4, 6), 16),
    g: parseInt(digits.slice(2, 4), 16),
    r: parseInt(digits.slice(0, 2), 16),
  };
};

/**
 * Parses any supported hex form into an {@link RgbaColor}. Returns `fallback`
 * (opaque black by default) for input that is not a 3/4/6/8-digit hex string.
 */
export const parseHexColor = (value: string, fallback: RgbaColor = BLACK): RgbaColor =>
  tryParseHexColor(value) ?? fallback;

/** Formats an {@link RgbaColor} as `#rrggbb`, or `#rrggbbaa` when `alpha` is set. */
export const formatHexColor = (color: RgbaColor, options?: FormatHexColorOptions): string => {
  const rgb = `#${toHexByte(color.r)}${toHexByte(color.g)}${toHexByte(color.b)}`;

  return options?.alpha ? `${rgb}${toHexByte(clampAlpha(color.a) * 255)}` : rgb;
};

/**
 * Splits a color into its `#rrggbb` part and a `[0, 1]` alpha, for consumers
 * that store the two separately (gradient stops used to do this inline).
 */
export const splitHexAlpha = (value: string): { hex: string; alpha: number } => {
  const color = parseHexColor(value);

  return { alpha: color.a, hex: formatHexColor(color) };
};

/** Recombines a `#rrggbb` and a `[0, 1]` alpha into `#rrggbbaa`. */
export const joinHexAlpha = (hex: string, alpha: number): string =>
  formatHexColor({ ...parseHexColor(hex), a: clampAlpha(alpha) }, { alpha: true });

/**
 * Normalizes any supported hex form to lowercase `#rrggbb`, or `#rrggbbaa` when
 * the input carries a non-opaque alpha. Used to compare values that may differ
 * only in case or shorthand. Unparseable input returns `fallback` verbatim, so
 * callers can pass through non-hex CSS colors untouched.
 */
export const normalizeHex = (value: string, fallback: string = '#000000'): string => {
  const color = tryParseHexColor(value);

  if (!color) {
    return fallback;
  }

  return formatHexColor(color, { alpha: color.a < 1 });
};
