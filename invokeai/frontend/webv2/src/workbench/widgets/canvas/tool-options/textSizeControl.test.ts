import { describe, expect, it } from 'vitest';

import { sliderPositionToTextSize, textSizeKeyboardStep, textSizeToSliderPosition, textSpecimen } from './TextOptions';

describe('text size slider scale', () => {
  it('round-trips whole pixel sizes across the track', () => {
    for (const size of [4, 12, 48, 100, 250, 600]) {
      expect(sliderPositionToTextSize(textSizeToSliderPosition(size))).toBe(size);
    }
  });

  it('spends most of the track on the sizes text is set at', () => {
    expect(textSizeToSliderPosition(48)).toBeGreaterThan(400);
    expect(textSizeToSliderPosition(48)).toBeLessThan(600);
    expect(textSizeToSliderPosition(4)).toBe(0);
    expect(textSizeToSliderPosition(600)).toBe(1000);
  });

  it('pins sizes beyond the track at its end and never returns a fraction', () => {
    expect(textSizeToSliderPosition(2000)).toBe(1000);
    expect(sliderPositionToTextSize(1000)).toBe(600);
    expect(sliderPositionToTextSize(-5)).toBe(4);
    expect(Number.isInteger(sliderPositionToTextSize(333))).toBe(true);
  });

  it('steps by whole pixels below 100 and by tens above, in both directions at the boundary', () => {
    expect(textSizeKeyboardStep(48, 1)).toBe(1);
    expect(textSizeKeyboardStep(99, 1)).toBe(1);
    expect(textSizeKeyboardStep(100, 1)).toBe(10);
    expect(textSizeKeyboardStep(100, -1)).toBe(1);
    expect(textSizeKeyboardStep(110, -1)).toBe(10);
  });
});

describe('textSpecimen', () => {
  it('shows the first non-empty line, trimmed and capped, else a neutral sample', () => {
    expect(textSpecimen(null)).toBe('Aa');
    expect(textSpecimen('   \n\t')).toBe('Aa');
    expect(textSpecimen('\n\n  Hello world  \nsecond')).toBe('Hello world');
    expect(textSpecimen('x'.repeat(60))).toHaveLength(40);
  });
});
