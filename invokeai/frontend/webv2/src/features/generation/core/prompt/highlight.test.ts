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

    it('marks an unknown wildcard as an error and a known one as recognised syntax', () => {
      const known = new Set(['colors']);
      const kindWithCatalog = (prompt: string, text: string) =>
        buildPromptHighlightSegments(prompt, { dynamicPrompts: true, knownWildcards: known }).find(
          (segment) => segment.text === text
        )?.kind;

      expect(kindWithCatalog('a __colors__ ball', '__colors__')).toBe('wildcard');
      expect(kindWithCatalog('a __nope__ ball', '__nope__')).toBe('error');
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
