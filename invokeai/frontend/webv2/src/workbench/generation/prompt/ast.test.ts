import { describe, expect, it } from 'vitest';

import { parsePrompt, serializePrompt, tokenizePrompt } from './ast';

describe('prompt AST', () => {
  const roundTrip = (prompt: string): string => serializePrompt(parsePrompt(prompt));

  it('tokenizes symbolic weights without treating hyphenated words as attention', () => {
    expect(tokenizePrompt('cat+').map((token) => token.type)).toEqual(['word', 'weight']);
    expect(tokenizePrompt('razor-sharp').map((token) => token.type)).toEqual(['word', 'punct', 'word']);
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
});
