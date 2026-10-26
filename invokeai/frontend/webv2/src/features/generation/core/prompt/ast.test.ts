import { describe, expect, it, vi } from 'vitest';

import { parsePrompt, serializePrompt, tokenizePrompt } from './ast';

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
