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
});
