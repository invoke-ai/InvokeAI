import corpus from '@features/generation/core/__fixtures__/animaPromptAttention.json';
import { describe, expect, it, vi } from 'vitest';

import { parseAnimaPrompt, parsePrompt, serializePrompt, tokenizePrompt } from './ast';

describe('prompt AST', () => {
  const roundTrip = (prompt: string): string => serializePrompt(parsePrompt(prompt));

  it('tokenizes symbolic weights without treating hyphenated words as attention', () => {
    expect(tokenizePrompt('cat+').map((token) => token.type)).toEqual(['word', 'weight']);
    expect(tokenizePrompt('razor-sharp').map((token) => token.type)).toEqual(['word', 'punct', 'word']);
  });

  it.each([
    ['(cat)-1', '-', '1'],
    ['(cat)+1', '+', '1'],
    ['(cat).5', 0.5, ''],
    ['(cat)2.', 2, ''],
    ['(cat)2next', 2, 'next'],
    ['(cat)++dog', '++', 'dog'],
  ] as const)('reads Anima group suffixes in %s', (prompt, weight, tail) => {
    const tokens = tokenizePrompt(prompt, 'anima');
    expect(tokens[3]).toMatchObject({ type: 'weight', value: weight });
    expect(tokens.slice(4).map((token) => ('value' in token ? token.value : ''))).toEqual(tail ? [tail] : []);
  });

  it.each([
    ['foo|bar+', ['foo|bar', '+']],
    ['foo{bar}+', ['foo{bar}', '+']],
    ['foo"bar+', ['foo"bar+']],
    ['foo.bar+', ['foo.bar+']],
    ['foo=bar+', ['foo=bar+']],
    ['foo++bar.baz', ['foo', '++', 'bar.baz']],
    ['foo++bar=2', ['foo', '++', 'bar=2']],
  ] as const)('uses backend bare-word boundaries in %s', (prompt, values) => {
    expect(tokenizePrompt(prompt, 'anima').map((token) => ('value' in token ? token.value : undefined))).toEqual(
      values
    );
  });

  it('keeps total substring reads linear across adjacent Anima attention runs', () => {
    const prompt = 'a++'.repeat(2000);
    const originalSlice = String.prototype.slice;
    let copiedChars = 0;
    const spy = vi.spyOn(String.prototype, 'slice').mockImplementation(function (
      this: string,
      start?: number,
      end?: number
    ) {
      if (this.length === prompt.length && start !== undefined) {
        copiedChars += (end ?? this.length) - start;
      }
      return originalSlice.call(this, start, end);
    });

    try {
      const tokens = tokenizePrompt(prompt, 'anima');
      expect(tokens).toHaveLength(4000);
      expect(tokens[0]).toMatchObject({ type: 'word', value: 'a' });
      expect(tokens.at(-1)).toMatchObject({ type: 'weight', value: '++' });
      expect(copiedChars).toBeLessThanOrEqual(prompt.length * 3);
    } finally {
      spy.mockRestore();
    }
  });

  it('parses prompt function argument ranges', () => {
    const node = parsePrompt("('one two', 'three four').and()")[0];

    expect(node?.type).toBe('prompt_function');

    if (node?.type !== 'prompt_function') {
      return;
    }

    expect(node.promptArgs).toHaveLength(2);
    expect(node.promptArgs[0]?.contentRange).toEqual({ start: 2, end: 9 });
    expect(node.promptArgs[1]?.contentRange).toEqual({ start: 13, end: 23 });
  });

  it.each([
    'a cat',
    '(a cat)',
    '(a cat)1.2',
    'cat+',
    'cat++',
    'cat-',
    '(hello world)+',
    '\\(medium\\)',
    'colored pencil \\(medium\\) (enhanced)',
    '<embedding_name>',
    'portrait \\(realistic\\) (high quality)1.2',
    "('one two', 'three four').and()",
    "('one', 'two three. four.').or()",
    "('one', 'two').blend(0.7, 0.3)",
    "('hello+', '(world)-').and()",
    "some text, ('a', 'b').and(), more text",
    "('a', 'b', 'c').blend(0.5, 0.3, 0.2)",
    '(one two, three four).and()',
    '(one two, three four).blend(0.7, 0.3)',
    '(a, b, c).blend(0.5, 0.3, 0.2)',
    'some text, (a, b).and(), more text',
    '(\u201cone\u201d, \u201ctwo\u201d).and()',
    '(\u2018one\u2019, \u2018two\u2019).or()',
    "('one',\n 'two',\n 'three').and()",
    'A bear \\(with razor-sharp teeth\\) in a forest.',
  ])('round-trips %s', (prompt) => {
    expect(roundTrip(prompt)).toBe(prompt);
  });

  it('normalizes whitespace between prompt function args and method names', () => {
    expect(roundTrip("('one', 'two')\n.and()")).toBe("('one', 'two').and()");
    expect(roundTrip('(one, two)\n.and()')).toBe('(one, two).and()');
  });

  it.each(['a ) tail', '(unfinished', 'a <unfinished', '< spaced embedding >', 'a > b', 'café+', '🌄+', '𐐀𐐁+'])(
    'preserves text and incomplete syntax: %s',
    (prompt) => {
      expect(roundTrip(prompt)).toBe(prompt);
    }
  );

  it('keeps unicode words and surrogate pairs together with their attention', () => {
    expect(tokenizePrompt('café+ 🌄+ 𐐀𐐁+').map((token) => token.type)).toEqual([
      'word',
      'weight',
      'whitespace',
      'word',
      'weight',
      'whitespace',
      'word',
      'weight',
    ]);
    expect(parsePrompt('🌄+')[0]).toMatchObject({ text: '🌄', attention: '+', range: { start: 0, end: 3 } });
  });
});

describe('Anima shared attention contract', () => {
  it.each(corpus.cases)('matches Compel weights for $input', ({ input, text, fragments }) => {
    const parsed = parseAnimaPrompt(input);
    const normalized = (value: string) => value.replace(/[ \t\r\n\v\f]+/g, ' ').replace(/^ +| +$/g, '');
    expect(normalized(parsed.text)).toBe(normalized(text));
    const actual = parsed.ranges.flatMap(({ start, end, weight }) =>
      Array.from(parsed.text.slice(start, end))
        .filter((char) => !' \t\r\n\v\f'.includes(char))
        .map((char) => ({ char, weight }))
    );
    const expected = fragments.flatMap(([text, weight]) =>
      Array.from(String(text))
        .filter((char) => !' \t\r\n\v\f'.includes(char))
        .map((char) => ({ char, weight: Number(weight) }))
    );
    expect(actual.map(({ char }) => char)).toEqual(expected.map(({ char }) => char));
    actual.forEach(({ weight }, index) => expect(weight).toBeCloseTo(expected[index]!.weight, 8));
  });

  it.each([
    ['red+(blue)2', 'red blue'],
    ['(red)2_blue', 'red _blue'],
    ['(foo.bar+)2', 'foo . bar'],
    ['red,blue+', 'red, blue'],
    ['(red)2,blue', 'red , blue'],
    ['(red)2.blue', 'red blue'],
    ['(cat)2\n\nnext', 'cat\n\nnext'],
    ['( red  blue )2', ' red  blue '],
    ['red  blue', 'red  blue'],
    ['foo.bar+', 'foo.bar+'],
    ['a(b)2', 'a(b)2'],
    ['cat)2', 'cat)2'],
    ['((red)2', '((red)2'],
  ])('pins fragment boundaries for %s', (input, expected) => {
    expect(parseAnimaPrompt(input).text).toBe(expected);
  });

  it('keeps malformed and deeply nested input iterative across parsing and painting tokens', () => {
    const unmatched = `${'('.repeat(20_000)}red)2`;
    expect(parseAnimaPrompt(unmatched).text).toBe(unmatched);
    expect(tokenizePrompt(unmatched, 'anima').some(({ type }) => type === 'weight')).toBe(false);
    const nested = `${'('.repeat(1500)}red${')'.repeat(1500)}2`;
    expect(parseAnimaPrompt(nested).text).toBe('red');
    expect(parsePrompt(nested, 'anima')[0]?.type).toBe('group');
  });
});

describe.each([
  { name: 'parentheses', prefix: '', inner: '=', suffix: '', copyBudget: 2 },
  { name: 'outer quote', prefix: '"', inner: '=', suffix: '"', copyBudget: 2 },
  { name: 'inner quote', prefix: '', inner: '"="', suffix: '', copyBudget: 2 },
  { name: 'discarded attention', prefix: '', inner: '(red)2 =', suffix: '2', copyBudget: 2 },
  { name: 'method boundary', prefix: '.and(', inner: '(red)2 =', suffix: ', dog+)', copyBudget: 2 },
  { name: 'malformed quote', prefix: '', inner: '"cat("', suffix: '65505', copyBudget: 2 },
  { name: 'decoded before malformed quote', prefix: '', inner: '"cat\\ndog" "cat("', suffix: '65505', copyBudget: 3 },
])('Anima invalid group rollback in $name', ({ prefix, inner, suffix, copyBudget }) => {
  it.each([1000, 2000, 4000])('copies source linearly at depth %i', (depth) => {
    const block = `${'('.repeat(depth)}${inner}${')'.repeat(depth)}`;
    const prompt = prefix + block + suffix;
    const originalSlice = String.prototype.slice;
    let copiedChars = 0;
    const spy = vi.spyOn(String.prototype, 'slice').mockImplementation(function (
      this: string,
      start?: number,
      end?: number
    ) {
      const value = originalSlice.call(this, start, end);
      if (this.length === prompt.length) {
        copiedChars += value.length;
      }
      return value;
    });
    let parsed: ReturnType<typeof parseAnimaPrompt>;
    try {
      parsed = parseAnimaPrompt(prompt);
    } finally {
      spy.mockRestore();
    }
    // Count requested characters rather than wall time or engine-specific string allocation.
    expect(copiedChars).toBeLessThanOrEqual(prompt.length * copyBudget);
    if (prefix === '.and(') {
      expect(parsed.text).toBe(`.and(${block} , dog)`);
      const dog = parsed.text.indexOf('dog');
      expect(parsed.ranges.find(({ start, end }) => start <= dog && end > dog)?.weight).toBeCloseTo(1.1);
      expect(parsed.ranges.find(({ start, end }) => start <= prefix.length && end > prefix.length)?.weight).toBe(1);
      expect(
        parsed.tokens.filter(({ type }) => type === 'weight').map((token) => 'value' in token && token.value)
      ).toEqual(['+']);
    } else {
      expect(parsed.text).toBe(prompt);
      expect(parsed.ranges.every(({ weight }) => weight === 1)).toBe(true);
      expect(parsed.tokens.some(({ type }) => type === 'weight')).toBe(false);
    }
  });
});

it.each(['2', '65505', '9'.repeat(310), '+'.repeat(10_000)])(
  'rolls rejected quoted targets back before validating suffix %s',
  (suffix) => {
    for (const child of ['"cat("', '"cat)"', String.raw`"cat\n("`]) {
      const prompt = `((${child})${suffix})2`;
      const parsed = parseAnimaPrompt(prompt);
      expect(parsed.text).toBe(prompt);
      expect(parsed.ranges.every(({ weight }) => weight === 1)).toBe(true);
      expect(parsed.tokens.some(({ type }) => type === 'weight')).toBe(false);
    }
  }
);

it.each([500, 1000, 2000])('recovers %i closed rejected quotes with linear source copies', (count) => {
  for (const separator of ['', ' ']) {
    const literal = Array(count)
      .fill(String.raw`"cat\fdog+"65505`)
      .join(separator);
    const prompt = `${literal} bird+`;
    const originalSlice = String.prototype.slice;
    let copiedChars = 0;
    const spy = vi.spyOn(String.prototype, 'slice').mockImplementation(function (
      this: string,
      start?: number,
      end?: number
    ) {
      const value = originalSlice.call(this, start, end);
      if (this.length === prompt.length) {
        copiedChars += value.length;
      }
      return value;
    });
    let parsed: ReturnType<typeof parseAnimaPrompt>;
    try {
      parsed = parseAnimaPrompt(prompt);
    } finally {
      spy.mockRestore();
    }
    expect(copiedChars).toBeLessThanOrEqual(prompt.length * 2);
    expect(parsed.text).toBe(`${literal} bird`);
    const bird = parsed.text.lastIndexOf('bird');
    expect(parsed.ranges.find(({ start, end }) => start <= bird && end > bird)?.weight).toBeCloseTo(1.1);
    expect(
      parsed.tokens.filter(({ type }) => type === 'weight').map((token) => 'value' in token && token.value)
    ).toEqual(['+']);
  }
});
