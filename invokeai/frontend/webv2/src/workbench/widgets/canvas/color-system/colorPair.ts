/**
 * Persist foreground/background preferences in canvas widget values, outside document history; pair changes never
 * rewrite existing content.
 */

import { formatHexColor, normalizeHex, parseHexColor } from '@platform/ui/color';

export type ActiveColorTarget = 'foreground' | 'background';

export interface ActiveColorPair {
  foreground: string;
  background: string;
}

/** Persisted keys inside the canvas widget's `state.values`. */
export const CANVAS_ACTIVE_COLORS_KEY = 'activeColors';
export const CANVAS_ACTIVE_COLOR_TARGET_KEY = 'activeColorTarget';
export const CANVAS_COLOR_PALETTE_KEY = 'colorPalette';

export const MAX_COLOR_PALETTE_SIZE = 24;

/** The classic reset pair; the foreground matches the brush default it seeds. */
export const DEFAULT_COLOR_PAIR: ActiveColorPair = { background: '#ffffff', foreground: '#000000' };

export const isActiveColorTarget = (value: unknown): value is ActiveColorTarget =>
  value === 'foreground' || value === 'background';

/** The pair and palette are opaque paint colors: parse or reject, and flatten any alpha byte. */
const toOpaqueHex = (value: unknown): string | null => {
  if (typeof value !== 'string' || normalizeHex(value, '') === '') {
    return null;
  }
  return formatHexColor(parseHexColor(value));
};

const readColor = (value: unknown, fallback: string): string => toOpaqueHex(value) ?? fallback;

/**
 * The persisted widget values are untyped; anything malformed falls back per
 * field, and an existing pair is preserved verbatim (idempotent migration).
 */
export const readActiveColorPair = (values: Record<string, unknown> | undefined): ActiveColorPair => {
  const raw = values?.[CANVAS_ACTIVE_COLORS_KEY];
  const pair = typeof raw === 'object' && raw !== null ? (raw as Partial<ActiveColorPair>) : {};
  return {
    background: readColor(pair.background, DEFAULT_COLOR_PAIR.background),
    foreground: readColor(pair.foreground, DEFAULT_COLOR_PAIR.foreground),
  };
};

export const readActiveColorTarget = (values: Record<string, unknown> | undefined): ActiveColorTarget => {
  const raw = values?.[CANVAS_ACTIVE_COLOR_TARGET_KEY];
  return isActiveColorTarget(raw) ? raw : 'foreground';
};

export const readColorPalette = (values: Record<string, unknown> | undefined): string[] => {
  const raw = values?.[CANVAS_COLOR_PALETTE_KEY];
  if (!Array.isArray(raw)) {
    return [];
  }
  const palette: string[] = [];
  for (const entry of raw) {
    const color = toOpaqueHex(entry);
    if (color === null || palette.length >= MAX_COLOR_PALETTE_SIZE || palette.includes(color)) {
      continue;
    }
    palette.push(color);
  }
  return palette;
};

export const swapColorPair = (pair: ActiveColorPair): ActiveColorPair => ({
  background: pair.foreground,
  foreground: pair.background,
});

export const withPairColor = (pair: ActiveColorPair, target: ActiveColorTarget, color: string): ActiveColorPair => ({
  ...pair,
  [target]: toOpaqueHex(color) ?? pair[target],
});

/** Appends like an artist's palette; a color already present stays put, and a full palette refuses. */
export const withPaletteColor = (palette: readonly string[], color: string): string[] => {
  const normalized = toOpaqueHex(color);
  if (normalized === null || palette.includes(normalized) || palette.length >= MAX_COLOR_PALETTE_SIZE) {
    return [...palette];
  }
  return [...palette, normalized];
};

export const withoutPaletteColor = (palette: readonly string[], color: string): string[] => {
  const normalized = toOpaqueHex(color);
  return palette.filter((entry) => entry !== normalized);
};

export const areColorPairsEqual = (a: ActiveColorPair, b: ActiveColorPair): boolean =>
  a.foreground === b.foreground && a.background === b.background;

export const areColorPalettesEqual = (a: readonly string[], b: readonly string[]): boolean =>
  a.length === b.length && a.every((entry, index) => entry === b[index]);
