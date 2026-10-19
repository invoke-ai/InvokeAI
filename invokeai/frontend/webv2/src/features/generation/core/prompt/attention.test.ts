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
    ['(a b)++', 'a', 'decrement', '(a b+)+'],
    ['(a b+)++', 'a', 'increment', '(a b)+++'],
    ['x (a+ b)+', 'x (a+', 'increment', '(x a++ b)+'],
    ['(a b)+', 'a', 'increment', '(a+ b)+'],
  ] as const)('adjusts and absorbs group selections: %s / %s', (prompt, selected, direction, expected) => {
    expect(adjust(prompt, selected, direction).prompt).toBe(expected);
  });

  it('keeps inherited symbolic attention when numeric style is preferred for new weights', () => {
    expect(adjust('(a b)+', 'a', 'increment', true).prompt).toBe('(a+ b)+');
  });

  it('preserves nested prompt functions when surrounding text changes', () => {
    expect(adjust("(x ('a', 'b').and())+", 'x', 'increment').prompt).toBe("(x+ ('a', 'b').and())+");
  });

  it('does not delete text after an unmatched closing parenthesis', () => {
    expect(adjust('a ) tail', 'a', 'increment').prompt).toBe('a+ ) tail');
  });

  it('keeps the selection local after absorbing an equal-weight neighbour', () => {
    const first = adjust('(a b+)++', 'a', 'increment');
    expect(first.prompt).toBe('(a b)+++');
    expect(first.prompt.slice(first.selectionStart, first.selectionEnd)).toBe('a');
    expect(adjustPromptAttention(first.prompt, first.selectionStart, first.selectionEnd, 'decrement').prompt).toBe(
      '(a b+)++'
    );
  });

  it('retains separate words when touching groups merge or become neutral', () => {
    expect(adjust('(a)1.1(b)1.2', 'a', 'increment').prompt).toBe('(a b)1.2');
    expect(adjust('(red)cat blue', 'blue', 'increment').prompt).toBe('red cat blue+');
    const result = adjust('(red)+(blue)+', 'red', 'decrement');
    expect(result).toEqual({ prompt: 'red blue+', selectionStart: 0, selectionEnd: 3 });
    expect(adjustPromptAttention(result.prompt, result.selectionStart, result.selectionEnd, 'decrement').prompt).toBe(
      'red- blue+'
    );
  });

  it.each([
    [1, '(red+ blue)+'],
    [9, '(red blue+)+'],
    [0, '(red blue)++'],
    [10, '(red blue)++'],
  ])('targets the adjacent word at content boundaries and the group outside them: caret %s', (caret, expected) => {
    expect(adjust('(red blue)+', [caret as number, caret as number], 'increment').prompt).toBe(expected);
  });

  it.each([
    ['a (b)1.1', 'a', 'increment', '(a b)1.1'],
    ['(a)1.1 b', 'b', 'increment', '(a b)1.1'],
    ['(a b)1.2', 'a', 'decrement', '(a)1.1 (b)1.2'],
    ['(a+ b)1.2', 'a+', 'increment', '(a)1.42 (b)1.2'],
    ['((a)1.2 b)+', 'a', 'increment', '(a)1.42 b+'],
    ['(a- b)+ c', 'c', 'increment', '(a)0.99 (b c)+'],
    ['(a)0 b', 'a', 'decrement', '(a)-0.1 b'],
    ['a (b c)+', 'a (', 'increment', '(a b c)+'],
    ['café landscape', 'café', 'increment', 'café+ landscape'],
    ['🌄 landscape', '🌄', 'increment', '🌄+ landscape'],
    ['🌄+ landscape', '🌄+', 'decrement', '🌄 landscape'],
    ['razor-sharp teeth', 'razor', 'increment', '(razor+)-sharp teeth'],
    ['(a)0.0000001 b', 'b', 'increment', '(a)0.0000001 b+'],
    ['a (unfinished', 'a', 'increment', 'a+ (unfinished'],
    ['a <unfinished', 'a', 'increment', 'a+ <unfinished'],
    ['a )1.20 tail', 'a', 'increment', 'a+ )1.20 tail'],
    ["(  'a',\n 'b'  ) .blend(0.70, 0.30)", 'a', 'increment', "(  'a+',\n 'b'  ) .blend(0.70, 0.30)"],
    ["(x ('a', 'b').and())+", 'a', 'increment', "(x ('a+', 'b').and())+"],
    ["(x ('a', 'b').and())2", 'a', 'increment', "(x ('(a)1.05', 'b').and())2"],
    ["(('a', 'b').and(), c).blend(0.5, 0.5)", 'a', 'increment', "(('a+', 'b').and(), c).blend(0.5, 0.5)"],
  ] as const)('handles weight and syntax boundaries: %s / %s', (prompt, selected, direction, expected) => {
    expect(adjust(prompt, selected, direction).prompt).toBe(expected);
  });

  it.each([' ', '\n', '().and()', "('a', 'b').blend(0.7, 0.3)"])(
    'leaves non-content selections alone: %s',
    (prompt) => {
      const selection: [number, number] = prompt.includes('blend') ? [prompt.indexOf('0.7'), prompt.length] : [0, 1];
      expect(adjust(prompt, selection, 'increment').prompt).toBe(prompt);
    }
  );

  it.each(['increment', 'decrement'] as const)('keeps long sequences of symbolic steps exact: %s', (direction) => {
    let result = { prompt: 'a b', selectionStart: 0, selectionEnd: 1 };
    for (let count = 0; count < 100; count++) {
      result = adjustPromptAttention(result.prompt, result.selectionStart, result.selectionEnd, direction);
    }
    expect(result.prompt).toBe(`a${(direction === 'increment' ? '+' : '-').repeat(100)} b`);
    for (let count = 0; count < 100; count++) {
      result = adjustPromptAttention(
        result.prompt,
        result.selectionStart,
        result.selectionEnd,
        direction === 'increment' ? 'decrement' : 'increment'
      );
    }
    expect(result).toEqual({ prompt: 'a b', selectionStart: 0, selectionEnd: 1 });
  });
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
