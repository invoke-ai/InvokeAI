import type { GradientStop } from '@workbench/canvas-engine/api';

/**
 * Pure stop-list math for the gradient stop strip. Stops are `#rrggbb` or
 * `#rrggbbaa` colors with offsets in [0, 1]; every operation returns a new
 * sorted list and never drops below two stops.
 */

const clamp01 = (value: number): number => Math.min(1, Math.max(0, value));

const parseChannel = (hex: string, index: number): number => {
  const parsed = Number.parseInt(hex.slice(1 + index * 2, 3 + index * 2), 16);
  return Number.isFinite(parsed) ? parsed : 0;
};

const parseRgba = (color: string): [number, number, number, number] => [
  parseChannel(color, 0),
  parseChannel(color, 1),
  parseChannel(color, 2),
  color.length >= 9 ? parseChannel(color, 3) : 255,
];

const toHexByte = (value: number): string =>
  Math.round(Math.min(255, Math.max(0, value)))
    .toString(16)
    .padStart(2, '0');

const sortStops = (stops: readonly GradientStop[]): GradientStop[] => [...stops].sort((a, b) => a.offset - b.offset);

/** The gradient's color at `offset`: linear interpolation between the flanking stops. */
export const sampleGradientColor = (stops: readonly GradientStop[], offset: number): string => {
  const sorted = sortStops(stops);
  const at = clamp01(offset);
  const first = sorted[0];
  const last = sorted[sorted.length - 1];
  if (!first || !last) {
    return '#000000ff';
  }
  if (at <= first.offset) {
    return first.color;
  }
  if (at >= last.offset) {
    return last.color;
  }
  const upperIndex = sorted.findIndex((stop) => stop.offset >= at);
  const upper = sorted[upperIndex]!;
  const lower = sorted[upperIndex - 1]!;
  const span = upper.offset - lower.offset;
  const mix = span <= 0 ? 0 : (at - lower.offset) / span;
  const a = parseRgba(lower.color);
  const b = parseRgba(upper.color);
  const blended = a.map((channel, i) => channel + (b[i]! - channel) * mix);
  return `#${blended.map(toHexByte).join('')}`;
};

/** Inserts a stop at `offset` with the gradient's own color there; returns the new list and the stop's index. */
export const insertStopAt = (
  stops: readonly GradientStop[],
  offset: number
): { stops: GradientStop[]; index: number } => {
  const at = clamp01(offset);
  const created: GradientStop = { color: sampleGradientColor(stops, at), offset: at };
  const next = sortStops([...stops, created]);
  return { index: next.indexOf(created), stops: next };
};

/** Moves one stop to `offset`; the list stays sorted, so the stop's index may change. */
export const moveStop = (
  stops: readonly GradientStop[],
  index: number,
  offset: number
): { stops: GradientStop[]; index: number } => {
  const target = stops[index];
  if (!target) {
    return { index, stops: [...stops] };
  }
  const moved: GradientStop = { ...target, offset: clamp01(offset) };
  const rest = stops.filter((_, i) => i !== index);
  const next = sortStops([...rest, moved]);
  return { index: next.indexOf(moved), stops: next };
};

/** Recolors one stop in place. */
export const recolorStop = (stops: readonly GradientStop[], index: number, color: string): GradientStop[] =>
  stops.map((stop, i) => (i === index ? { ...stop, color } : stop));

/** Removes one stop; refused (returns the input) when only two remain. */
export const removeStop = (stops: readonly GradientStop[], index: number): GradientStop[] =>
  stops.length <= 2 ? [...stops] : stops.filter((_, i) => i !== index);

/** The strip's own preview: a horizontal CSS gradient of the stops. */
export const stopsToCssGradient = (stops: readonly GradientStop[]): string => {
  const parts = sortStops(stops).map((stop) => `${stop.color} ${(stop.offset * 100).toFixed(1)}%`);
  return `linear-gradient(90deg, ${parts.join(', ')})`;
};
