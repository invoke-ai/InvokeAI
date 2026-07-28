import { describe, expect, it } from 'vitest';

import { buildPromptHighlightSegments, type PromptHighlightKind } from './highlight';

const segmentsByText = (prompt: string, text: string) =>
  buildPromptHighlightSegments(prompt).filter((segment) => segment.text === text || segment.text.includes(text));

const kindForText = (prompt: string, text: string): PromptHighlightKind | undefined =>
  segmentsByText(prompt, text)[0]?.kind;

describe('prompt highlight segments', () => {
  it('highlights symbolic and numeric attention separately', () => {
    const segments = buildPromptHighlightSegments('cat+ (dog)1.2');

    expect(segments).toContainEqual({ kind: 'attention', range: { start: 3, end: 4 }, text: '+' });
    expect(segments).toContainEqual({ kind: 'attentionNumeric', range: { start: 10, end: 13 }, text: '1.2' });
  });

  it('highlights embeddings as a semantic range', () => {
    const prompt = 'use <embedding_name> now';

    expect(kindForText(prompt, '<embedding_name>')).toBe('embedding');
  });

  it('highlights escaped parentheses as literal prompt syntax', () => {
    const segments = buildPromptHighlightSegments('literal \\(medium\\)');

    expect(segments).toContainEqual({ kind: 'escapedParen', range: { start: 8, end: 10 }, text: '\\(' });
    expect(segments).toContainEqual({ kind: 'escapedParen', range: { start: 16, end: 18 }, text: '\\)' });
  });

  it('highlights prompt function args and method tails cosmetically', () => {
    const prompt = "('one two', 'three four').and()";

    expect(kindForText(prompt, 'one two')).toBe('promptFunctionArg');
    expect(kindForText(prompt, 'three four')).toBe('promptFunctionArg');
    expect(kindForText(prompt, '.and()')).toBe('promptFunctionMethod');
  });

  it('keeps attention syntax inside prompt function args higher priority than arg background', () => {
    const prompt = "('one+', 'two').and()";

    expect(kindForText(prompt, '+')).toBe('attention');
  });

  it('marks unmatched parentheses as cosmetic errors', () => {
    expect(kindForText('(unclosed', '(')).toBe('error');
    expect(kindForText('extra)', ')')).toBe('error');
  });

  it('does not mark hyphenated words as attention', () => {
    const prompt = 'razor-sharp teeth';

    expect(kindForText(prompt, '-')).toBe('punctuation');
  });

  describe('dynamic prompt syntax', () => {
    const dynamicKindForText = (prompt: string, text: string): PromptHighlightKind | undefined =>
      buildPromptHighlightSegments(prompt, { dynamicPrompts: true }).find((segment) => segment.text === text)?.kind;

    it('is off by default, so surfaces that never expand see plain text', () => {
      const prompt = 'a {red|green} ball';

      expect(kindForText(prompt, '{')).toBe('text');
      expect(kindForText(prompt, '|')).toBe('punctuation');
    });

    it('marks variant braces, separators and weights when enabled', () => {
      expect(dynamicKindForText('a {red|green} ball', '{')).toBe('variantBrace');
      expect(dynamicKindForText('a {red|green} ball', '|')).toBe('variantSeparator');
      expect(dynamicKindForText('a {red|green} ball', '}')).toBe('variantBrace');
      expect(dynamicKindForText('{2::red|green}', '2::')).toBe('variantWeight');
      expect(dynamicKindForText('{1-2$$red|green}', '1-2$$')).toBe('variantRange');
      expect(dynamicKindForText('a __color__ ball', '__color__')).toBe('wildcard');
      expect(dynamicKindForText(`\${colour} ball`, `\${colour}`)).toBe('promptVariable');
    });

    it('marks a sampler prefix and a variable’s assignment operator', () => {
      expect(dynamicKindForText('{~red|green}', '~')).toBe('variantSampler');
      expect(dynamicKindForText('{@red|green}', '@')).toBe('variantSampler');
      expect(dynamicKindForText(`\${lens=85mm}`, '=')).toBe('promptVariableOperator');
      // A bare reference has no operator to mark, and stays one flat span.
      expect(dynamicKindForText(`\${lens}`, `\${lens}`)).toBe('promptVariable');
    });

    // The comment wins over the error the unknown wildcard would otherwise be:
    // upstream strips the comment before it ever looks the name up.
    it('dims a comment over everything inside it', () => {
      const known = new Set(['colors']);
      const segments = buildPromptHighlightSegments('a cat # __nope__ {unclosed', {
        dynamicPrompts: true,
        knownWildcards: known,
      });

      expect(segments.filter((segment) => segment.range.start >= 6).map((segment) => segment.kind)).toEqual([
        'comment',
      ]);
    });

    it('marks an unknown wildcard as an error and a known one as recognised syntax', () => {
      const known = new Set(['colors']);
      const kindWithCatalog = (prompt: string, text: string) =>
        buildPromptHighlightSegments(prompt, { dynamicPrompts: true, knownWildcards: known }).find(
          (segment) => segment.text === text
        )?.kind;

      expect(kindWithCatalog('a __colors__ ball', '__colors__')).toBe('wildcard');
      expect(kindWithCatalog('a __nope__ ball', '__nope__')).toBe('error');
      // A glob resolves against the catalog rather than being flagged for not
      // being a name of its own.
      expect(kindWithCatalog('a __colo*__ ball', '__colo*__')).toBe('wildcard');
      expect(kindWithCatalog('a __nope/*__ ball', '__nope/*__')).toBe('error');
      // Without a catalog nothing is known to be missing, so neither is an error.
      expect(dynamicKindForText('a __nope__ ball', '__nope__')).toBe('wildcard');
    });

    it('marks an unbalanced brace as an error, like an unbalanced parenthesis', () => {
      expect(dynamicKindForText('a {red green', '{')).toBe('error');
      expect(dynamicKindForText('a red} green', '}')).toBe('error');
    });

    it('leaves attention syntax inside a variant intact', () => {
      expect(dynamicKindForText('{red+|green}', '+')).toBe('attention');
    });
  });
});

// The annotation lookup used to filter and sort every annotation for every
// token. Dynamic prompts made that bite: a variant-heavy prompt now carries an
// annotation per brace, separator, weight, sampler, variable and wildcard, so
// the annotation count became proportional to the token count. This ran to
// 469ms on the prompt below, synchronously inside a render, on every keystroke.
describe('buildPromptHighlightSegments on a large prompt', () => {
  // eslint-disable-next-line no-template-curly-in-string -- `${v=1}` is dynamic-prompts syntax, not interpolation
  const prompt = '{a|b|c} __colours__ ${v=1} '.repeat(700);
  const options = { dynamicPrompts: true, knownWildcards: new Set(['colours']) };

  it('covers the whole prompt exactly once', () => {
    const segments = buildPromptHighlightSegments(prompt, options);

    expect(segments.map((segment) => segment.text).join('')).toBe(prompt);
  });

  it('stays well inside a frame', () => {
    const started = performance.now();

    buildPromptHighlightSegments(prompt, options);

    expect(performance.now() - started).toBeLessThan(150);
  });
});
