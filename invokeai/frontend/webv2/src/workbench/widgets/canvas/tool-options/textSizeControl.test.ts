import { describe, expect, it } from 'vitest';

import { textSizeKeyboardStep, textSpecimen } from './TextOptions';

describe('text size keyboard step', () => {
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
