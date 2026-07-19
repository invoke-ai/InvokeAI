import { describe, expect, it } from 'vitest';

import type { PromptAttentionDirection } from './attention';

import { adjustPromptAttention } from './attention';

const adjust = (
  prompt: string,
  selected: string | [number, number],
  direction: PromptAttentionDirection,
  preferNumericAttentionStyle = false
) => {
  const range =
    typeof selected === 'string'
      ? ([prompt.indexOf(selected), prompt.indexOf(selected) + selected.length] as [number, number])
      : selected;

  return adjustPromptAttention(prompt, range[0], range[1], direction, preferNumericAttentionStyle);
};

describe('prompt attention adjustment', () => {
  it.each([
    ['hello world', 'hello', 'increment', 'hello+ world'],
    ['hello world', 'hello', 'decrement', 'hello- world'],
    ['hello+ world', 'hello+', 'increment', 'hello++ world'],
    ['hello+ world', 'hello+', 'decrement', 'hello world'],
    ['hello- world', 'hello-', 'decrement', 'hello-- world'],
    ['hello- world', 'hello-', 'increment', 'hello world'],
  ] as const)('adjusts a single word: %s %s', (prompt, selected, direction, expected) => {
    expect(adjust(prompt, selected, direction).prompt).toBe(expected);
  });

  it.each([
    ['hello world', [0, 11], 'increment', '(hello world)+'],
    ['hello world', [0, 11], 'decrement', '(hello world)-'],
    ['one, two', [3, 3], 'increment', 'one+, two'],
    ['one, two', [5, 5], 'increment', 'one, two+'],
  ] as const)('adjusts selections and cursor boundaries: %s', (prompt, selected, direction, expected) => {
    expect(adjust(prompt, selected as [number, number], direction).prompt).toBe(expected);
  });

  it.each([
    ['(hello world)+', [13, 14], 'increment', '(hello world)++'],
    ['(hello world)+', [0, 14], 'decrement', 'hello world'],
    ['(a b)+', [1, 2], 'increment', '(a+ b)+'],
    ['(a b)+ c', [3, 8], 'increment', '(a b+ c)+'],
    ['(a b)+ c', [3, 8], 'decrement', 'a+ b c-'],
  ] as const)('preserves and splits existing groups: %s', (prompt, selected, direction, expected) => {
    expect(adjust(prompt, selected as [number, number], direction).prompt).toBe(expected);
  });

  it('preserves the adjusted selection range', () => {
    const result = adjust('(a b)+', [1, 2], 'increment');

    expect(result.prompt).toBe('(a+ b)+');
    expect(result.prompt.slice(result.selectionStart, result.selectionEnd)).toBe('a+');
  });

  it.each([
    ['(masterpiece)1.3', [0, 16], 'increment', '(masterpiece)1.4'],
    ['(masterpiece)1.3', [0, 16], 'decrement', '(masterpiece)1.2'],
    ['(sunny midday light)1.15', [0, 24], 'increment', '(sunny midday light)1.25'],
  ] as const)('adjusts explicit numeric weights additively: %s', (prompt, selected, direction, expected) => {
    expect(adjust(prompt, selected as [number, number], direction).prompt).toBe(expected);
  });

  it('does not corrupt unselected numeric weights', () => {
    const prompt = '(masterpiece)1.3, best quality, (sunny midday light)1.15, clear sky';
    const result = adjust(prompt, 'clear sky', 'increment');

    expect(result.prompt).toContain('(masterpiece)1.3');
    expect(result.prompt).toContain('(sunny midday light)1.15');
    expect(result.prompt).toContain('(clear sky)+');
    expect(result.prompt).not.toMatch(/\d\.\d{5,}/);
  });

  it.each([
    ["('hello world', 'other').and()", 'hello', 'increment', "('hello+ world', 'other').and()"],
    ["('a', 'hello world').or()", 'hello world', 'increment', "('a', '(hello world)+').or()"],
    ["('one two', 'three four').blend(0.7, 0.3)", 'two', 'increment', "('one two+', 'three four').blend(0.7, 0.3)"],
    ['(hello world, foo bar).and()', 'hello', 'increment', '(hello+ world, foo bar).and()'],
    [
      '(\u201chello world\u201d, \u201cother\u201d).and()',
      'hello',
      'increment',
      '(\u201chello+ world\u201d, \u201cother\u201d).and()',
    ],
  ] as const)('adjusts inside prompt function args: %s', (prompt, selected, direction, expected) => {
    expect(adjust(prompt, selected, direction).prompt).toBe(expected);
  });

  it('adjusts both prompt function args when a selection spans an argument separator', () => {
    const prompt = "('one two', 'three four').and()";
    const start = prompt.indexOf('two');
    const end = prompt.indexOf('three') + 'three'.length;
    const result = adjustPromptAttention(prompt, start, end, 'increment');

    expect(result.prompt).toBe("('one two+', 'three+ four').and()");
    expect(result.prompt.slice(result.selectionStart, result.selectionEnd)).toContain('two+');
    expect(result.prompt.slice(result.selectionStart, result.selectionEnd)).toContain('three+');
  });

  it('supports project numeric attention style for new weights', () => {
    expect(adjust('hello world', 'hello', 'increment', true).prompt).toBe('(hello)1.1 world');
    expect(adjust('hello world', [0, 11], 'decrement', true).prompt).toBe('(hello world)0.9');
    expect(adjust("('one two', 'three four').and()", 'one', 'increment', true).prompt).toBe(
      "('(one)1.1 two', 'three four').and()"
    );
  });

  it('keeps hyphenated words intact when adjacent text is adjusted', () => {
    expect(adjust('razor-sharp teeth', 'teeth', 'increment').prompt).toBe('razor-sharp teeth+');
  });
});
