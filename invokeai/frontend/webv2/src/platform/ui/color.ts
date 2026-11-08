/** An RGBA color. `r`/`g`/`b` are `[0, 255]` integers; `a` is a `[0, 1]` float. */
export interface RgbaColor {
  r: number;
  g: number;
  b: number;
  a: number;
}

export interface FormatHexColorOptions {
  alpha?: boolean;
}

export const BLACK: RgbaColor = { a: 1, b: 0, g: 0, r: 0 };

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

const expandNibble = (nibble: string): number => parseInt(`${nibble}${nibble}`, 16);

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

export const parseHexColor = (value: string, fallback: RgbaColor = BLACK): RgbaColor =>
  tryParseHexColor(value) ?? fallback;

export const formatHexColor = (color: RgbaColor, options?: FormatHexColorOptions): string => {
  const rgb = `#${toHexByte(color.r)}${toHexByte(color.g)}${toHexByte(color.b)}`;

  return options?.alpha ? `${rgb}${toHexByte(clampAlpha(color.a) * 255)}` : rgb;
};

export const splitHexAlpha = (value: string): { hex: string; alpha: number } => {
  const color = parseHexColor(value);

  return { alpha: color.a, hex: formatHexColor(color) };
};

export const joinHexAlpha = (hex: string, alpha: number): string =>
  formatHexColor({ ...parseHexColor(hex), a: clampAlpha(alpha) }, { alpha: true });

export const normalizeHex = (value: string, fallback: string = '#000000'): string => {
  const color = tryParseHexColor(value);

  if (!color) {
    return fallback;
  }

  return formatHexColor(color, { alpha: color.a < 1 });
};
