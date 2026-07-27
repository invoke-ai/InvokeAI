import { describe, expect, it } from 'vitest';

import { scanDynamicPromptSyntax } from './dynamicPromptSyntax';

const annotate = (prompt: string) =>
  scanDynamicPromptSyntax(prompt)
    .sort((left, right) => left.range.start - right.range.start)
    .map(({ kind, range }) => [kind, prompt.slice(range.start, range.end)]);

describe('scanDynamicPromptSyntax', () => {
  it('annotates braces and separators of a variant', () => {
    expect(annotate('a {red|green|blue} ball')).toEqual([
      ['variantBrace', '{'],
      ['variantSeparator', '|'],
      ['variantSeparator', '|'],
      ['variantBrace', '}'],
    ]);
  });

  it('leaves a pipe outside a variant alone', () => {
    expect(annotate('a | b')).toEqual([]);
  });

  it('annotates value weights', () => {
    expect(annotate('{2::red|1.5::green}')).toEqual([
      ['variantBrace', '{'],
      ['variantWeight', '2::'],
      ['variantSeparator', '|'],
      ['variantWeight', '1.5::'],
      ['variantBrace', '}'],
    ]);
  });

  it('annotates count and range prefixes, including a custom separator', () => {
    expect(annotate('{2$$a|b}')).toEqual([
      ['variantBrace', '{'],
      ['variantRange', '2$$'],
      ['variantSeparator', '|'],
      ['variantBrace', '}'],
    ]);
    expect(annotate('{1-3$$a|b}')).toEqual([
      ['variantBrace', '{'],
      ['variantRange', '1-3$$'],
      ['variantSeparator', '|'],
      ['variantBrace', '}'],
    ]);
    expect(annotate('{2$$ and $$a|b}')).toEqual([
      ['variantBrace', '{'],
      ['variantRange', '2$$ and $$'],
      ['variantSeparator', '|'],
      ['variantBrace', '}'],
    ]);
  });

  it('handles nested variants', () => {
    expect(annotate('{a|{b|c}}')).toEqual([
      ['variantBrace', '{'],
      ['variantSeparator', '|'],
      ['variantBrace', '{'],
      ['variantSeparator', '|'],
      ['variantBrace', '}'],
      ['variantBrace', '}'],
    ]);
  });

  it('flags an unclosed brace', () => {
    expect(annotate('a {red|green ball')).toEqual([
      ['error', '{'],
      ['variantSeparator', '|'],
    ]);
  });

  it('flags a stray closing brace', () => {
    expect(annotate('a red} ball')).toEqual([['error', '}']]);
  });

  it('annotates wildcards', () => {
    expect(annotate('a __color__ __animals/dogs__')).toEqual([
      ['wildcard', '__color__'],
      ['wildcard', '__animals/dogs__'],
    ]);
  });

  it('annotates a variable reference as one span, without descending into it', () => {
    expect(annotate(`\${colour} ball`)).toEqual([['promptVariable', `\${colour}`]]);
    expect(annotate(`\${colour=red|green}`)).toEqual([['promptVariable', `\${colour=red|green}`]]);
  });

  it('skips escaped braces', () => {
    expect(annotate('a \\{literal\\} brace')).toEqual([]);
  });

  it('returns nothing for a plain prompt', () => {
    expect(annotate('a photo of a cat, highly detailed')).toEqual([]);
  });
});
