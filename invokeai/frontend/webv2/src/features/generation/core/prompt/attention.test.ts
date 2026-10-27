import { describe, expect, it } from 'vitest';

import type { PromptAttentionDirection } from './attention';

import { parseAnimaPrompt } from './ast';
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
    ['blue-green+', 'blue-green++'],
    ['hair!+', 'hair!++'],
    ['a+b+', 'a+b++'],
    ['foo/bar+', 'foo/bar++'],
    ['foo:bar+', 'foo:bar++'],
    ['a?b+', 'a?b++'],
    ['word_+', 'word_++'],
    ['foo,bar+', 'foo,bar++'],
  ])('uses Anima word boundaries for %s', (prompt, expected) => {
    const caret = Math.floor(prompt.length / 2);
    expect(adjustPromptAttention(prompt, caret, caret, 'increment', false, 'anima').prompt).toBe(expected);
  });

  it('does not rewrite unsupported Compel method calls for Anima', () => {
    const prompt = "('red', 'blue').and()";
    const caret = prompt.indexOf('and');
    expect(adjustPromptAttention(prompt, caret, caret, 'increment', false, 'anima').prompt).toBe(prompt);
  });

  it.each([
    ['blue-green+ .and()', 'blue-green++ .and()'],
    ['blue-green+ .blend(0.5, 0.5)', 'blue-green++ .blend(0.5, 0.5)'],
    ["blue-green+ ('red', 'blue').and()", "blue-green++ ('red', 'blue').and()"],
    ["blue-green+ ('red', 'blue').blend(0.5, 0.5)", "blue-green++ ('red', 'blue').blend(0.5, 0.5)"],
    ['.and() blue-green+', '.and() blue-green++'],
  ])('preserves unsupported calls while editing nearby Anima attention in %s', (prompt, expected) => {
    const caret = prompt.indexOf('blue') + 3;
    expect(adjustPromptAttention(prompt, caret, caret, 'increment', false, 'anima').prompt).toBe(expected);
  });

  it('edits Anima attention inside the group before an unsupported method without changing its tail', () => {
    const prompt = '(red+, blue).and() blue-green+';
    const caret = prompt.indexOf('red') + 1;
    expect(adjustPromptAttention(prompt, caret, caret, 'increment', false, 'anima').prompt).toBe(
      '(red++, blue).and() blue-green+'
    );
  });

  it.each([0, '(red, blue'.length])(
    'adjusts the whole group from delimiter caret %i before an unsupported method',
    (caret) => {
      const prompt = '(red, blue).and()';
      expect(adjustPromptAttention(prompt, caret, caret, 'increment', false, 'anima').prompt).toBe(
        '(red, blue)+.and()'
      );
    }
  );

  it('adjusts the second weighted word after an adjacent symbolic suffix for Anima', () => {
    const prompt = 'foo++bar+';
    const caret = prompt.indexOf('bar') + 1;
    const result = adjustPromptAttention(prompt, caret, caret, 'increment', false, 'anima');

    expect(result.prompt).toBe('foo++bar++');
  });

  it('keeps a zero Anima group at zero when decreasing attention', () => {
    const prompt = '(cat)0';
    const caret = prompt.indexOf('cat') + 1;
    expect(adjustPromptAttention(prompt, caret, caret, 'decrement', false, 'anima').prompt).toBe(prompt);
    expect(adjustPromptAttention(prompt, caret, caret, 'increment', false, 'anima').prompt).toBe('(cat)0.1');
  });

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

describe('Anima source-preserving edits', () => {
  const weightAt = (prompt: string, word: string): number => {
    const parsed = parseAnimaPrompt(prompt);
    const start = parsed.text.indexOf(word);
    expect(start).toBeGreaterThanOrEqual(0);
    const weights = parsed.ranges
      .filter((range) => range.start < start + word.length && range.end > start)
      .map(({ weight }) => weight);
    expect(weights.length).toBeGreaterThan(0);
    weights.forEach((weight) => expect(weight).toBeCloseTo(weights[0]!, 10));
    return weights[0]!;
  };
  const change = (prompt: string, target: string | number, direction: PromptAttentionDirection = 'increment') => {
    const caret = typeof target === 'number' ? target : prompt.indexOf(target) + 1;
    const edited = adjustPromptAttention(prompt, caret, caret, direction, false, 'anima');
    expect(parseAnimaPrompt(edited.prompt).text).toBe(parseAnimaPrompt(prompt).text);
    return edited.prompt;
  };

  it.each([
    ['red\\+ cat', 'red\\+', 1],
    ['foo\\(bar\\)+ cat', 'foo(bar)', 1.1],
    ['(cat)! dog', 'cat', 1],
    ['(cat)😀 dog', 'cat', 1],
    ['red\\.blue+ cat', 'red\\.blue', 1.1],
  ] as const)('keeps unselected source and weights in %s', (prompt, preserved, weight) => {
    const target = prompt.endsWith('dog') ? 'dog' : 'cat';
    const edited = change(prompt, target);
    expect(edited.slice(0, prompt.lastIndexOf(' '))).toBe(prompt.slice(0, prompt.lastIndexOf(' ')));
    expect(weightAt(edited, preserved)).toBeCloseTo(weight);
    expect(weightAt(edited, target)).toBeCloseTo(1.1);
  });

  it('targets only the next word at an adjacent group boundary', () => {
    const edited = change('(cat)2next', 6);
    expect(weightAt(edited, 'cat')).toBe(2);
    expect(weightAt(edited, 'next')).toBeCloseTo(1.1);
  });

  it.each(['red', 'cat'])('steps effective numeric attention through method contexts: %s', (target) => {
    const prompt = '((red, blue).and() cat)2';
    const edited = change(prompt, target);
    expect(edited).toContain('.and()');
    for (const word of ['red', 'blue', 'cat']) {
      expect(weightAt(edited, word)).toBeCloseTo(word === target ? 2.1 : 2, 10);
    }
  });

  it.each(['.and(dog+)', '.and(dog+', '.and(.blend(dog+))', '.and((dog)+', '(red).and((dog)+, cat)'])(
    'edits active attention inside complete, nested and partial methods: %s',
    (prompt) => {
      const edited = change(prompt, 'dog');
      expect(weightAt(edited, 'dog')).toBeCloseTo(1.21);
      expect(edited).toContain('.and(');
    }
  );

  it.each(['red.attend(2)', 'red .attend (2)', '"red blue" .attend(2)'])(
    'steps supported .attend numerics: %s',
    (prompt) => {
      const edited = change(prompt, 'red');
      expect(weightAt(edited, 'red')).toBeCloseTo(2.1, 10);
      if (prompt.includes('blue')) {
        expect(weightAt(edited, 'blue')).toBe(2);
      }
    }
  );

  it('steps each numeric descendant from a selected group delimiter', () => {
    const edited = change('(cat+ dog)2', 0);
    expect(weightAt(edited, 'cat')).toBeCloseTo(2.3, 10);
    expect(weightAt(edited, 'dog')).toBeCloseTo(2.1, 10);
  });

  it('multiplies opposite symbolic signs rather than cancelling them', () => {
    const edited = change('(red- blue)+', 'red');
    expect(weightAt(edited, 'red')).toBeCloseTo(1.09, 10);
    expect(weightAt(edited, 'blue')).toBeCloseTo(1.1, 10);
  });

  it('can increase whole zero groups but leaves impossible partial edits unchanged', () => {
    expect(weightAt(change('(cat)0', 'cat'), 'cat')).toBeCloseTo(0.1);
    expect(change('(cat)0', 'cat', 'decrement')).toBe('(cat)0');
    expect(change('(red blue)0', 'red')).toBe('(red blue)0');
    const whole = change('(red blue)0', 0);
    expect(weightAt(whole, 'red')).toBeCloseTo(0.1);
    expect(weightAt(whole, 'blue')).toBeCloseTo(0.1);
  });

  it.each(['(cat)65504', `cat${'+'.repeat(116)}`, '((cat)65504 blue)0.5', '(cat+ dog)59549.05'])(
    'refuses edits beyond a local or composed float16 bound: %s',
    (prompt) => {
      expect(change(prompt, 'cat')).toBe(prompt);
    }
  );

  it.each(['foo.bar+', 'foo=bar+', 'foo"bar+', '++x+', 'a(b)2', '((red)2', 'cat)2'])(
    'does not activate syntax in a literal fallback while editing %s',
    (prompt) => {
      expect(change(prompt, 1)).toBe(prompt);
    }
  );

  it('keeps nested parsing and editing iterative', () => {
    const prompt = `${'('.repeat(1500)}cat dog${')'.repeat(1500)}2`;
    const edited = change(prompt, 'cat');
    expect(weightAt(edited, 'cat')).toBeCloseTo(2.1);
    expect(weightAt(edited, 'dog')).toBe(2);
  });
});

it.each([
  ['red+(blue)2', 'red', 1, 'blue', 2],
  ['red++blue+', 'red', 1.1, 'blue', 1.1],
  ['red+-', 'red', 1, '-', 1],
  ['(red)+-', 'red', 1, '-', 1],
] as const)(
  'keeps adjacent fragments and literal signs when decreasing %s',
  (prompt, target, expected, other, otherWeight) => {
    const caret = prompt.indexOf(target) + 1;
    const edited = adjustPromptAttention(prompt, caret, caret, 'decrement', false, 'anima').prompt;
    expect(edited).not.toBe(prompt);
    const parsed = parseAnimaPrompt(edited);
    expect(parsed.text).toBe(parseAnimaPrompt(prompt).text);
    for (const [word, weight] of [
      [target, expected],
      [other, otherWeight],
    ] as const) {
      const index = parsed.text.indexOf(word);
      expect(parsed.ranges.find(({ start, end }) => start <= index && end > index)?.weight).toBeCloseTo(weight, 10);
    }
  }
);

it.each(['(red).and(dog+)', '(red).blend(dog+)'])(
  'keeps the unsupported marker plain when its dot is selected in %s',
  (prompt) => {
    const caret = prompt.indexOf('.');
    expect(adjustPromptAttention(prompt, caret, caret, 'increment', false, 'anima').prompt).toBe(prompt);
  }
);

it('keeps unselected literal method marker weights when editing their only neighbouring word', () => {
  const prompt = '(cat .and())2';
  const caret = prompt.indexOf('cat') + 1;
  const edited = adjustPromptAttention(prompt, caret, caret, 'increment', false, 'anima').prompt;
  const parsed = parseAnimaPrompt(edited);
  expect(parsed.text).toBe(parseAnimaPrompt(prompt).text);
  const method = parsed.text.indexOf('.and');
  expect(parsed.ranges.find(({ start, end }) => start <= method && end > method)?.weight).toBe(2);
  expect(parsed.ranges.find(({ start, end }) => start <= 0 && end > 0)?.weight).toBeCloseTo(2.1);
});

it('steps mixed numeric descendants and literal method punctuation from a group delimiter', () => {
  const prompt = '(cat+ .and(dog))2';
  const edited = adjustPromptAttention(prompt, 0, 0, 'increment', false, 'anima').prompt;
  const parsed = parseAnimaPrompt(edited);
  expect(parsed.text).toBe(parseAnimaPrompt(prompt).text);
  for (const [word, weight] of [
    ['cat', 2.3],
    ['.and', 2.1],
    ['dog', 2.1],
  ] as const) {
    const index = parsed.text.indexOf(word);
    expect(parsed.ranges.find(({ start, end }) => start <= index && end > index)?.weight).toBeCloseTo(weight, 10);
  }
});

it.each(['(cat)2(next)2', '(cat)2"next"2'])(
  'targets the next group at a touching delimiter boundary in %s',
  (prompt) => {
    const edited = adjustPromptAttention(prompt, 6, 6, 'increment', false, 'anima').prompt;
    const parsed = parseAnimaPrompt(edited);
    expect(parsed.text).toBe('cat next');
    expect(parsed.ranges.find(({ start, end }) => start <= 0 && end > 0)?.weight).toBe(2);
    expect(parsed.ranges.find(({ start, end }) => start <= 4 && end > 4)?.weight).toBeCloseTo(2.1, 10);
  }
);

describe('Anima attention over deep, wide groups', () => {
  it.each([320, 1280, 6400])('adjusts a width and depth of %i through the owning interface', (size) => {
    const prompt = `${'('.repeat(size)}${'cat '.repeat(size)}${')'.repeat(size)}`;
    const result = adjustPromptAttention(prompt, size, size + 4 * size - 1, 'increment', false, 'anima');
    const parsed = parseAnimaPrompt(result.prompt);
    expect(parsed.text).toBe('cat '.repeat(size));
    expect(result.prompt.startsWith('('.repeat(size))).toBe(true);
    expect(result.prompt.endsWith(')'.repeat(size))).toBe(true);
    let weightedChars = 0;
    for (const { start, end, weight } of parsed.ranges) {
      const content = parsed.text.slice(start, end).replaceAll(' ', '');
      if (content) {
        expect(weight).toBeCloseTo(1.1, 10);
        weightedChars += content.length;
      }
    }
    expect(weightedChars).toBe(3 * size);
  });

  it('keeps an unselected sibling at its inherited numeric weight', () => {
    const size = 1280;
    const prompt = `${'('.repeat(size)}${'cat '.repeat(size)}dog${')'.repeat(size)}2`;
    const result = adjustPromptAttention(prompt, size, size + 4 * size - 1, 'increment', false, 'anima');
    const parsed = parseAnimaPrompt(result.prompt);
    expect(parsed.text).toBe(`${'cat '.repeat(size)}dog`);
    for (const { start, end, weight } of parsed.ranges) {
      const content = parsed.text.slice(start, end).trim();
      if (content) {
        expect(weight).toBeCloseTo(content === 'dog' ? 2 : 2.1, 10);
      }
    }
  });
});

it.each(['n', 't', 'r'])(
  'keeps quoted \\%s source offsets and unselected effective weights while editing dog',
  (escape) => {
    for (const direction of ['increment', 'decrement'] as const) {
      for (const backslashes of [1, 2, 3]) {
        const prompt = `"cat${'\\'.repeat(backslashes)}${escape}dog+"2`;
        const caret = prompt.indexOf('dog');
        const result = adjustPromptAttention(prompt, caret, caret, direction, false, 'anima');
        expect(result.prompt).not.toBe(prompt);
        expect(result.prompt.slice(0, caret)).toBe(prompt.slice(0, caret));
        const parsed = parseAnimaPrompt(result.prompt);
        expect(parsed.text).toBe(parseAnimaPrompt(prompt).text);
        for (const word of ['cat', 'dog']) {
          const index = parsed.text.indexOf(word);
          expect(parsed.ranges.find(({ start, end }) => start <= index && end > index)?.weight).toBeCloseTo(
            word === 'cat' ? 2 : direction === 'increment' ? 2.3 : 2.1,
            10
          );
        }
      }
    }
  }
);

it('keeps an outside-quote backslash-n in the same weighted word', () => {
  const prompt = String.raw`cat\ndog+`;
  const caret = prompt.indexOf('dog');
  const result = adjustPromptAttention(prompt, caret, caret, 'increment', false, 'anima');
  const parsed = parseAnimaPrompt(result.prompt);
  expect(parsed.text).toBe(String.raw`cat\ndog`);
  expect(parsed.ranges.every(({ weight }) => Math.abs(weight - 1.21) < 1e-12)).toBe(true);
});

it('keeps decoded whitespace and nested numeric attention when editing a quoted descendant', () => {
  const prompt = String.raw`"cat\n(dog+)2"2`;
  const caret = prompt.indexOf('dog') + 1;
  const result = adjustPromptAttention(prompt, caret, caret, 'increment', false, 'anima');
  const parsed = parseAnimaPrompt(result.prompt);
  expect(result.prompt.slice(0, caret - 1)).toBe(prompt.slice(0, caret - 1));
  expect(parsed.text).toBe('cat\ndog');
  expect(parsed.ranges.find(({ start, end }) => start === 0 && end > 0)?.weight).toBe(2);
  const dog = parsed.text.indexOf('dog');
  expect(parsed.ranges.find(({ start, end }) => start <= dog && end > dog)?.weight).toBeCloseTo(4.5, 10);
});

it.each([
  String.raw`"cat\fdog+"65505 bird+`,
  '"cat("2 bird+',
  '"cat)"2 bird+',
  String.raw`"cat\fdog+"2 "dog"2 bird+`,
  String.raw`.and("cat\fdog+"2, bird+)`,
  String.raw`.and(("cat\fdog+"2)65505, bird+)`,
  String.raw`.and(.blend("cat\fdog+"2, bird+))`,
])('preserves rejected quote source and cleaned text when adjusting a following target: %s', (prompt) => {
  const caret = prompt.indexOf('bird') + 1;
  for (const direction of ['increment', 'decrement'] as const) {
    const result = adjustPromptAttention(prompt, caret, caret, direction, false, 'anima');
    expect(result.prompt).not.toBe(prompt);
    expect(result.prompt.slice(0, prompt.indexOf('bird'))).toBe(prompt.slice(0, prompt.indexOf('bird')));
    const parsed = parseAnimaPrompt(result.prompt);
    expect(parsed.text).toBe(parseAnimaPrompt(prompt).text);
    const bird = parsed.text.indexOf('bird');
    expect(parsed.ranges.find(({ start, end }) => start <= bird && end > bird)?.weight).toBeCloseTo(
      direction === 'increment' ? 1.21 : 1,
      10
    );
    expect(parsed.ranges.find(({ start, end }) => start <= 0 && end > 0)?.weight).toBe(1);
  }
});
