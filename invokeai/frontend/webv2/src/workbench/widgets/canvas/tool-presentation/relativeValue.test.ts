import { describe, expect, it } from 'vitest';

import { parseRelativeInput, resolveRelativeValue } from './relativeValue';

describe('relative values', () => {
  it('resolves operators typed or pasted against the current value, and refuses incomplete ones', () => {
    const resolve = (text: string) => {
      const parsed = parseRelativeInput(text);
      return parsed && resolveRelativeValue(400, parsed.operator, Number.parseFloat(parsed.operand));
    };
    expect(resolve('50%')).toBe(200);
    expect(resolve('*2')).toBe(800);
    expect(resolve('÷4')).toBe(100);
    expect(resolve('*')).toBeNull();
    expect(resolve('%')).toBeNull();
    expect(resolve('/0')).toBeNull();
    expect(parseRelativeInput('120')).toBeNull();
  });
});
