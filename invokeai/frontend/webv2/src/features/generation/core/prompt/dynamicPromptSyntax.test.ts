import { scanWildcardReferences } from '@features/generation/core/dynamicPrompts';
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

  // Regression: globs, sampler overrides and parameters are all valid, and all
  // read as ordinary text — or worse, as errors — before this.
  it('annotates the looser reference forms the backend accepts', () => {
    expect(
      annotate('__artists/*__ __artist?__ __animals/[dc]ogs__ __~colours__ __@colours__ __outfit(mood=warm)__')
    ).toEqual([
      ['wildcard', '__artists/*__'],
      ['wildcard', '__artist?__'],
      ['wildcard', '__animals/[dc]ogs__'],
      ['wildcard', '__~colours__'],
      ['wildcard', '__@colours__'],
      ['wildcard', '__outfit(mood=warm)__'],
    ]);
  });

  it('highlights exactly the references returned by the shared scanner', () => {
    const prompt = '__artist?__ snake__case __~outfit(mood=warm)__ __animals/[!c]ogs__';
    const highlighted = scanDynamicPromptSyntax(prompt)
      .filter(({ kind }) => kind === 'wildcard')
      .map(({ range }) => range);

    expect(highlighted).toEqual(scanWildcardReferences(prompt).map(({ range }) => range));
  });

  it('flags a reference no known wildcard answers to', () => {
    const known = new Set(['colours', 'animals/dogs']);
    const scan = (prompt: string) => scanDynamicPromptSyntax(prompt, known).map(({ kind }) => kind);

    expect(scan('__colours__')).toEqual(['wildcard']);
    expect(scan('__nope__')).toEqual(['error']);
    // A glob is known when anything matches it, and `*` crosses `/` as it does
    // on the backend.
    expect(scan('__animals/*__')).toEqual(['wildcard']);
    expect(scan('__*s__')).toEqual(['wildcard']);
    expect(scan('__nope/*__')).toEqual(['error']);
    // The sampler and the parameters are not part of the name being looked up.
    expect(scan('__~colours__')).toEqual(['wildcard']);
    expect(scan('__colours(mood=warm)__')).toEqual(['wildcard']);
  });

  it('leaves snake__case alone', () => {
    expect(annotate('a snake__case word')).toEqual([]);
  });

  it('annotates a variable reference as one span, without descending into it', () => {
    expect(annotate(`\${colour} ball`)).toEqual([['promptVariable', `\${colour}`]]);
  });

  it('marks the operator that makes a variable an assignment', () => {
    expect(annotate(`\${colour=red|green}`)).toEqual([
      ['promptVariable', `\${colour=red|green}`],
      ['promptVariableOperator', '='],
    ]);
    expect(annotate(`\${colour=!{red|green}}`)).toEqual([
      ['promptVariable', `\${colour=!{red|green}}`],
      ['promptVariableOperator', '=!'],
    ]);
    expect(annotate(`\${colour:crimson}`)).toEqual([
      ['promptVariable', `\${colour:crimson}`],
      ['promptVariableOperator', ':'],
    ]);
  });

  // A `:` in the assigned value belongs to the value, not to this variable.
  it('marks only the first operator in a variable', () => {
    expect(annotate(`\${caption=a:b:c}`)).toEqual([
      ['promptVariable', `\${caption=a:b:c}`],
      ['promptVariableOperator', '='],
    ]);
  });

  it('annotates the sampler that opens a variant', () => {
    expect(annotate('{~red|green}')).toEqual([
      ['variantBrace', '{'],
      ['variantSampler', '~'],
      ['variantSeparator', '|'],
      ['variantBrace', '}'],
    ]);
    expect(annotate('{@red|green}')).toEqual([
      ['variantBrace', '{'],
      ['variantSampler', '@'],
      ['variantSeparator', '|'],
      ['variantBrace', '}'],
    ]);
  });

  // The sampler binds tighter than the count: `{2$$~a|b}` has `~a` as a value.
  it('takes the sampler before the count, and only there', () => {
    expect(annotate('{~2$$red|green}')).toEqual([
      ['variantBrace', '{'],
      ['variantSampler', '~'],
      ['variantRange', '2$$'],
      ['variantSeparator', '|'],
      ['variantBrace', '}'],
    ]);
    expect(annotate('{2$$~red|green}')).toEqual([
      ['variantBrace', '{'],
      ['variantRange', '2$$'],
      ['variantSeparator', '|'],
      ['variantBrace', '}'],
    ]);
  });

  it('dims a comment, and everything it swallows', () => {
    expect(annotate('a cat # {unclosed __nope__')).toEqual([['comment', '# {unclosed __nope__']]);
  });

  // `\#` is not an escape upstream — it comments from the `#` and leaves the
  // backslash in the prompt — so the scanner must not treat it as one either.
  it('comments from a backslash-escaped hash too', () => {
    expect(annotate('a \\# b')).toEqual([['comment', '# b']]);
  });

  it('ends a comment at the newline', () => {
    expect(annotate('a # note\n{red|green}')).toEqual([
      ['comment', '# note'],
      ['variantBrace', '{'],
      ['variantSeparator', '|'],
      ['variantBrace', '}'],
    ]);
  });

  it('skips escaped braces', () => {
    expect(annotate('a \\{literal\\} brace')).toEqual([]);
  });

  it('returns nothing for a plain prompt', () => {
    expect(annotate('a photo of a cat, highly detailed')).toEqual([]);
  });
});
