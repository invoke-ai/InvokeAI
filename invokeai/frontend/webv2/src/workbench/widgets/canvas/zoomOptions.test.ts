import { ZOOM_SNAP_CANDIDATES } from '@workbench/canvas-engine/math/snapping';
import { describe, expect, it } from 'vitest';

import { formatZoomPercent, zoomMenuOptions } from './zoomOptions';

describe('formatZoomPercent', () => {
  it('renders whole percentages', () => {
    expect(formatZoomPercent(1)).toBe('100%');
    expect(formatZoomPercent(0.25)).toBe('25%');
    expect(formatZoomPercent(2)).toBe('200%');
  });

  it('rounds to the nearest whole percent', () => {
    expect(formatZoomPercent(0.333)).toBe('33%');
    expect(formatZoomPercent(0.675)).toBe('68%');
  });
});

describe('zoomMenuOptions', () => {
  it('covers every snap candidate', () => {
    const options = zoomMenuOptions();
    expect(options).toHaveLength(ZOOM_SNAP_CANDIDATES.length);
    expect(new Set(options.map((option) => option.value))).toEqual(new Set(ZOOM_SNAP_CANDIDATES));
  });

  it('is ordered from largest to smallest zoom', () => {
    const values = zoomMenuOptions().map((option) => option.value);
    expect(values).toEqual([...values].sort((a, b) => b - a));
    expect(values[0]).toBe(Math.max(...ZOOM_SNAP_CANDIDATES));
  });

  it('pairs each value with its formatted label', () => {
    for (const option of zoomMenuOptions()) {
      expect(option.label).toBe(formatZoomPercent(option.value));
    }
  });
});
