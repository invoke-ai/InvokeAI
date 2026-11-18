import type { ServerOutputRow } from './transcript';

/** A palette index 0..255 (0-15 are the theme's ANSI colors), a validated `#rrggbb`, or the view's own default. */
export type OutputColor = { kind: 'palette'; index: number } | { kind: 'rgb'; hex: string } | { kind: 'default' };

export interface SegmentStyle {
  readonly fg?: OutputColor;
  readonly bg?: OutputColor;
  readonly bold: boolean;
  readonly dim: boolean;
  readonly italic: boolean;
  readonly underline: boolean;
}

export interface RowSegment {
  readonly text: string;
  readonly style: SegmentStyle | null;
}

const PALETTE_INDEX = /^(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)$/;
const RGB = /^#[0-9a-f]{6}$/i;

/** Anything outside the two documented forms is dropped, never passed to CSS. */
export const parseOutputColor = (value: unknown): OutputColor | undefined => {
  if (typeof value !== 'string') {
    return undefined;
  }

  if (PALETTE_INDEX.test(value)) {
    return { index: Number(value), kind: 'palette' };
  }

  return RGB.test(value) ? { hex: value.toLowerCase(), kind: 'rgb' } : undefined;
};

const CUBE_LEVELS = [0, 95, 135, 175, 215, 255];

const toHex = (red: number, green: number, blue: number): string =>
  `#${[red, green, blue].map((channel) => channel.toString(16).padStart(2, '0')).join('')}`;

/** The xterm 256-color table above the 16 theme colors. */
export const extendedPaletteHex = (index: number): string => {
  if (index >= 232) {
    const level = 8 + (index - 232) * 10;
    return toHex(level, level, level);
  }

  const cube = index - 16;
  return toHex(CUBE_LEVELS[Math.floor(cube / 36)]!, CUBE_LEVELS[Math.floor(cube / 6) % 6]!, CUBE_LEVELS[cube % 6]!);
};

const isUnit = (value: unknown): value is number => typeof value === 'number' && Number.isInteger(value) && value >= 0;

const toStyle = (run: ServerOutputRow['styles'][number]): SegmentStyle | null => {
  let fg = parseOutputColor(run.fg);
  let bg = parseOutputColor(run.bg);

  if (run.inverse === true) {
    [fg, bg] = [bg ?? { kind: 'default' }, fg ?? { kind: 'default' }];
  }

  const style: SegmentStyle = {
    bg,
    bold: run.bold === true,
    dim: run.dim === true,
    fg,
    italic: run.italic === true,
    underline: run.underline === true,
  };

  return fg || bg || style.bold || style.dim || style.italic || style.underline ? style : null;
};

/**
 * Splits a row's text at its style runs. Runs are clamped to the text; a run that overlaps an earlier one or runs
 * backwards is ignored, so every character appears exactly once.
 */
export const segmentRow = (row: ServerOutputRow): RowSegment[] => {
  const segments: RowSegment[] = [];
  let position = 0;

  for (const run of row.styles) {
    if (!isUnit(run.offset) || !isUnit(run.length) || run.offset < position) {
      continue;
    }

    const start = Math.min(run.offset, row.text.length);
    const end = Math.min(start + run.length, row.text.length);
    const style = toStyle(run);

    if (end === start) {
      continue;
    }

    if (start > position) {
      segments.push({ style: null, text: row.text.slice(position, start) });
    }

    segments.push({ style, text: row.text.slice(start, end) });
    position = end;
  }

  if (position < row.text.length || segments.length === 0) {
    segments.push({ style: null, text: row.text.slice(position) });
  }

  return segments;
};
