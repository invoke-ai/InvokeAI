import type { GradientStop } from '@workbench/canvas-engine/api';

import { describe, expect, it } from 'vitest';

import {
  insertStopAt,
  moveStop,
  recolorStop,
  removeStop,
  sampleGradientColor,
  stopsToCssGradient,
} from './gradientStops';

const BW: GradientStop[] = [
  { color: '#000000ff', offset: 0 },
  { color: '#ffffffff', offset: 1 },
];

describe('gradient stop math', () => {
  it('samples the interpolated color, clamped at the ends', () => {
    expect(sampleGradientColor(BW, 0.5)).toBe('#808080ff');
    expect(sampleGradientColor(BW, -1)).toBe('#000000ff');
    expect(sampleGradientColor(BW, 2)).toBe('#ffffffff');
    // Alpha interpolates too, and a #rrggbb stop counts as opaque.
    const translucent: GradientStop[] = [
      { color: '#ff000000', offset: 0 },
      { color: '#ff0000', offset: 1 },
    ];
    expect(sampleGradientColor(translucent, 0.5)).toBe('#ff000080');
  });

  it('inserts a stop with the gradient color at that offset, sorted', () => {
    const { index, stops } = insertStopAt(BW, 0.25);
    expect(index).toBe(1);
    expect(stops).toHaveLength(3);
    expect(stops[1]).toEqual({ color: '#404040ff', offset: 0.25 });
    expect(stops.map((stop) => stop.offset)).toEqual([0, 0.25, 1]);
  });

  it('moves a stop, keeping the list sorted and tracking its index across neighbours', () => {
    const three = insertStopAt(BW, 0.25).stops;
    const moved = moveStop(three, 1, 0.9);
    expect(moved.stops.map((stop) => stop.offset)).toEqual([0, 0.9, 1]);
    expect(moved.index).toBe(1);
    const swapped = moveStop(three, 0, 0.95);
    expect(swapped.stops.map((stop) => stop.offset)).toEqual([0.25, 0.95, 1]);
    expect(swapped.index).toBe(1);
    expect(moveStop(three, 1, 7).stops[2]!.offset).toBe(1);
  });

  it('recolors in place and refuses to remove below two stops', () => {
    expect(recolorStop(BW, 0, '#112233ff')[0]!.color).toBe('#112233ff');
    expect(removeStop(BW, 0)).toEqual(BW);
    const three = insertStopAt(BW, 0.5).stops;
    expect(removeStop(three, 1)).toEqual(BW);
  });

  it('renders the css preview in offset order', () => {
    const unsorted: GradientStop[] = [
      { color: '#ffffffff', offset: 1 },
      { color: '#000000ff', offset: 0 },
    ];
    expect(stopsToCssGradient(unsorted)).toBe('linear-gradient(90deg, #000000ff 0.0%, #ffffffff 100.0%)');
  });
});
