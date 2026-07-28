import { describe, expect, it } from 'vitest';

import {
  getWildcardNameError,
  hasDynamicPromptSyntax,
  sanitizeDynamicPromptsConfig,
  sanitizeMaxPrompts,
} from './dynamicPrompts';

describe('hasDynamicPromptSyntax', () => {
  it('detects a variant anywhere in the prompt', () => {
    expect(hasDynamicPromptSyntax('a {red|green} ball')).toBe(true);
    expect(hasDynamicPromptSyntax('{a}')).toBe(true);
    expect(hasDynamicPromptSyntax('multi\nline {a|b}')).toBe(true);
  });

  it('detects a wildcard reference', () => {
    expect(hasDynamicPromptSyntax('a __colors__ ball')).toBe(true);
    expect(hasDynamicPromptSyntax('a __animals/dogs__ ball')).toBe(true);
    expect(hasDynamicPromptSyntax('a __artists/*__ ball')).toBe(true);
    expect(hasDynamicPromptSyntax('a __~colors__ ball')).toBe(true);
    expect(hasDynamicPromptSyntax('a __outfit(mood=warm)__ ball')).toBe(true);
  });

  // A comment is only stripped by going through the expander, and upstream has
  // no escape for `#`, so any prompt containing one has to take the trip.
  it('detects a comment', () => {
    expect(hasDynamicPromptSyntax('a red ball # for now')).toBe(true);
    expect(hasDynamicPromptSyntax('a red ball \\# for now')).toBe(true);
  });

  it('ignores prompts with neither a variant nor a wildcard', () => {
    expect(hasDynamicPromptSyntax('a red ball')).toBe(false);
    expect(hasDynamicPromptSyntax('unclosed { brace')).toBe(false);
    // Not a reference: a name may not start or end with an underscore, so this
    // cannot round-trip through the `__` delimiters.
    expect(hasDynamicPromptSyntax('snake__case word')).toBe(false);
  });
});

describe('sanitizeMaxPrompts', () => {
  it('clamps to the backend bounds and falls back on garbage', () => {
    expect(sanitizeMaxPrompts(50)).toBe(50);
    expect(sanitizeMaxPrompts(0)).toBe(1);
    expect(sanitizeMaxPrompts(99_999)).toBe(10_000);
    expect(sanitizeMaxPrompts(12.6)).toBe(13);
    expect(sanitizeMaxPrompts('many')).toBe(100);
    expect(sanitizeMaxPrompts(undefined)).toBe(100);
  });
});

describe('sanitizeDynamicPromptsConfig', () => {
  it('returns null for non-object values', () => {
    expect(sanitizeDynamicPromptsConfig(null)).toBeNull();
    expect(sanitizeDynamicPromptsConfig('nope')).toBeNull();
  });

  it('defaults every unusable field', () => {
    expect(sanitizeDynamicPromptsConfig({})).toEqual({
      combinatorial: true,
      maxPrompts: 100,
      sampleSeed: 0,
      seedBehaviour: 'per-iteration',
    });
    expect(
      sanitizeDynamicPromptsConfig({ combinatorial: false, maxPrompts: 7, sampleSeed: 42, seedBehaviour: 'per-image' })
    ).toEqual({
      combinatorial: false,
      maxPrompts: 7,
      sampleSeed: 42,
      seedBehaviour: 'per-image',
    });
    expect(sanitizeDynamicPromptsConfig({ seedBehaviour: 'PER_PROMPT' })?.seedBehaviour).toBe('per-iteration');
  });
});

describe('getWildcardNameError', () => {
  it('accepts the names the backend accepts', () => {
    expect(getWildcardNameError('colors')).toBeNull();
    expect(getWildcardNameError('animals/dogs')).toBeNull();
    expect(getWildcardNameError('sci-fi_props')).toBeNull();
    expect(getWildcardNameError('  colors  ')).toBeNull();
  });

  it('rejects the names the backend rejects', () => {
    expect(getWildcardNameError('')).toBe('empty');
    expect(getWildcardNameError('   ')).toBe('empty');
    expect(getWildcardNameError('colors list')).toBe('invalid');
    // A leading or trailing underscore would run into the `__` delimiters.
    expect(getWildcardNameError('_colors')).toBe('invalid');
    expect(getWildcardNameError('colors_')).toBe('invalid');
    expect(getWildcardNameError('a'.repeat(129))).toBe('tooLong');
    expect(getWildcardNameError('a'.repeat(128))).toBeNull();
  });

  it('reports a name the user already owns', () => {
    expect(getWildcardNameError('colors', new Set(['colors']))).toBe('taken');
    expect(getWildcardNameError('shades', new Set(['colors']))).toBeNull();
  });
});
