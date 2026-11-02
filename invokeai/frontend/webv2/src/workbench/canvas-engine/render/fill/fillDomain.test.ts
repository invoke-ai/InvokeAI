import {
  exceedsFillPixelCap,
  MAX_FILL_PIXELS,
  resolveFillDomain,
} from '@workbench/canvas-engine/render/fill/fillDomain';
import { describe, expect, it } from 'vitest';

const frame = { height: 512, width: 512, x: 0, y: 0 };
const domain = (input: Partial<Parameters<typeof resolveFillDomain>[0]>) =>
  resolveFillDomain({ clipToFrame: false, contentBounds: null, frame, selectionBounds: null, ...input });

describe('resolveFillDomain', () => {
  it('fills the frame when nothing is sampled', () => {
    expect(domain({})).toEqual(frame);
    expect(domain({ contentBounds: { height: 0, width: 0, x: 900, y: 900 } })).toEqual(frame);
  });

  it('follows content past the frame', () => {
    expect(domain({ contentBounds: { height: 100, width: 1000, x: -200, y: 300 } })).toEqual({
      height: 512,
      width: 1000,
      x: -200,
      y: 0,
    });
  });

  it('rounds fractional content bounds outwards to whole pixels', () => {
    expect(domain({ contentBounds: { height: 10, width: 10, x: -3.5, y: 505.2 } })).toEqual({
      height: 516,
      width: 516,
      x: -4,
      y: 0,
    });
  });

  it('stays inside the selection bounds', () => {
    const contentBounds = { height: 2000, width: 2000, x: -500, y: -500 };
    expect(domain({ contentBounds, selectionBounds: { height: 50, width: 40, x: 600, y: 10.5 } })).toEqual({
      height: 51,
      width: 40,
      x: 600,
      y: 10,
    });
  });

  it('is null when the selection misses the domain', () => {
    expect(domain({ selectionBounds: { height: 10, width: 10, x: 600, y: 600 } })).toBeNull();
  });

  it('clips to the frame when asked', () => {
    const contentBounds = { height: 2000, width: 2000, x: -500, y: -500 };
    expect(domain({ clipToFrame: true, contentBounds })).toEqual(frame);
    expect(
      domain({ clipToFrame: true, contentBounds, selectionBounds: { height: 100, width: 100, x: 480, y: -20 } })
    ).toEqual({ height: 80, width: 32, x: 480, y: 0 });
  });
});

describe('exceedsFillPixelCap', () => {
  it('allows exactly the cap and refuses one row more', () => {
    expect(exceedsFillPixelCap({ height: 4096, width: 4096, x: 0, y: 0 })).toBe(false);
    expect(exceedsFillPixelCap({ height: 4097, width: 4096, x: -9, y: 3 })).toBe(true);
    expect(exceedsFillPixelCap({ height: 1, width: MAX_FILL_PIXELS + 1, x: 0, y: 0 })).toBe(true);
  });
});
