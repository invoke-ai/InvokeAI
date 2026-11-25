import { describe, expect, it } from 'vitest';

import { getGuidanceSliderMax } from './guidanceSlider';

describe('getGuidanceSliderMax', () => {
  it('keeps the 10-wide track for floors below it', () => {
    expect(getGuidanceSliderMax(1, null)).toBe(10);
    expect(getGuidanceSliderMax(1, 20)).toBe(10);
    expect(getGuidanceSliderMax(1, 6)).toBe(6);
    expect(getGuidanceSliderMax(0, 100)).toBe(10);
  });

  it('spans a usable range above a floor at or beyond the track, containing the FLUX Fill default', () => {
    const max = getGuidanceSliderMax(25, null);

    expect(max).toBe(35);
    expect(30).toBeGreaterThanOrEqual(25);
    expect(30).toBeLessThanOrEqual(max);
    expect(getGuidanceSliderMax(10, null)).toBe(20);
  });

  it('never crosses the input ceiling or drops below the floor', () => {
    expect(getGuidanceSliderMax(25, 30)).toBe(30);
    expect(getGuidanceSliderMax(25, 25)).toBe(25);
  });
});
