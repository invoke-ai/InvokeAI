import type { Rect } from 'features/controlLayers/store/types';
import { describe, expect, it } from 'vitest';

import { rectToIdeogram4Bbox } from './buildIdeogram4Prompt';

// The JSON caption assembly moved to the backend `ideogram4_caption_builder` node (so dynamic prompts /
// batching vary the encoded caption); its behavior is covered by tests/backend/ideogram4/test_caption.py.
// Only the bbox normalization stays in the frontend (it needs the canvas generation bbox).

describe('rectToIdeogram4Bbox', () => {
  const genBbox: Rect = { x: 0, y: 0, width: 1024, height: 1024 };

  it('normalizes a region rect to [y_min, x_min, y_max, x_max] in 0-1000', () => {
    // Region occupying the right half, vertically centered band.
    const regionRect: Rect = { x: 512, y: 256, width: 512, height: 512 };
    expect(rectToIdeogram4Bbox(regionRect, genBbox)).toEqual([250, 500, 750, 1000]);
  });

  it('accounts for a generation bbox not anchored at the origin', () => {
    const offsetBbox: Rect = { x: 100, y: 200, width: 1000, height: 500 };
    const regionRect: Rect = { x: 600, y: 200, width: 500, height: 250 };
    // x: (600-100)/1000=0.5 -> 500, (1100-100)/1000=1.0 -> 1000
    // y: (200-200)/500=0 -> 0, (450-200)/500=0.5 -> 500
    expect(rectToIdeogram4Bbox(regionRect, offsetBbox)).toEqual([0, 500, 500, 1000]);
  });

  it('clamps out-of-bounds regions to the 0-1000 range', () => {
    const regionRect: Rect = { x: -200, y: -200, width: 2048, height: 2048 };
    expect(rectToIdeogram4Bbox(regionRect, genBbox)).toEqual([0, 0, 1000, 1000]);
  });

  it('returns zeros for a degenerate (zero-size) generation bbox', () => {
    const regionRect: Rect = { x: 10, y: 10, width: 10, height: 10 };
    expect(rectToIdeogram4Bbox(regionRect, { x: 0, y: 0, width: 0, height: 0 })).toEqual([0, 0, 0, 0]);
  });
});
