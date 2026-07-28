import { describe, expect, it } from 'vitest';

import {
  getWildcardNameError,
  getWildcardValuesError,
  hasDynamicPromptSyntax,
  matchesKnownWildcard,
  normalizeWildcardValues,
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

describe('matchesKnownWildcard', () => {
  const CATALOG = new Set(['colours', 'colours/warm', 'animals/dogs', 'moods']);

  it('looks a plain path up directly', () => {
    expect(matchesKnownWildcard('colours', CATALOG)).toBe(true);
    expect(matchesKnownWildcard('nope', CATALOG)).toBe(false);
  });

  it('resolves globs, including across the `/` separator', () => {
    expect(matchesKnownWildcard('colo*', CATALOG)).toBe(true);
    expect(matchesKnownWildcard('*s', CATALOG)).toBe(true);
    expect(matchesKnownWildcard('animals/*', CATALOG)).toBe(true);
    // `*` is an unanchored run of anything, `/` included — as on the backend.
    expect(matchesKnownWildcard('*/dogs', CATALOG)).toBe(true);
    expect(matchesKnownWildcard('*', CATALOG)).toBe(true);
    expect(matchesKnownWildcard('nope/*', CATALOG)).toBe(false);
    expect(matchesKnownWildcard('colours/*', new Set(['colours']))).toBe(false);
  });

  it('anchors both ends, so the two cannot share characters', () => {
    // `ab*ba` needs four characters; matching `ab` and `ba` onto the same `b`
    // is what a naive prefix/suffix check gets wrong.
    expect(matchesKnownWildcard('ab*ba', new Set(['aba']))).toBe(false);
    expect(matchesKnownWildcard('ab*ba', new Set(['abba']))).toBe(true);
  });

  it('takes the interior runs in order', () => {
    expect(matchesKnownWildcard('a*b*c', new Set(['axbxc']))).toBe(true);
    expect(matchesKnownWildcard('a*b*c', new Set(['axcxb']))).toBe(false);
    // Adjacent stars collapse: `**` means what `*` means.
    expect(matchesKnownWildcard('a**c', new Set(['abc']))).toBe(true);
  });

  // Regression: translating the path to `^a.*.*.*…$` backtracks exponentially,
  // and the path is whatever the user has typed so far. Twelve stars against one
  // name used to take about a minute.
  it('does not backtrack on a path full of stars', () => {
    const names = new Set(Array.from({ length: 50 }, (_, index) => `colours/warm-autumn-palette-${index}`));
    const started = performance.now();

    expect(matchesKnownWildcard(`${'*'.repeat(24)}q`, names)).toBe(false);
    expect(performance.now() - started).toBeLessThan(100);
  });
});

describe('normalizeWildcardValues', () => {
  // What the editor stores has to be what an export writes and a re-import
  // reads back. Blank lines used to survive into the catalog and then vanish on
  // the next round trip, so the two disagreed.
  it('drops blank lines and trims the rest', () => {
    expect(normalizeWildcardValues(['red', '', '  green  ', '   ', 'blue'])).toEqual(['red', 'green', 'blue']);
  });

  it('leaves an already-clean list alone', () => {
    expect(normalizeWildcardValues(['red', 'green'])).toEqual(['red', 'green']);
  });

  // Verified against dynamicprompts itself: a value of `poster #1` expands to
  // `poster `, and `#ff0000 glow` expands to nothing. There is no escape for it,
  // so a stored `#` is text that silently never reaches the model.
  it('drops a comment, and the value that is nothing but one', () => {
    expect(normalizeWildcardValues(['poster #1', 'red # a colour', '#ff0000 glow', 'blue'])).toEqual([
      'poster',
      'red',
      'blue',
    ]);
  });
});

describe('getWildcardValuesError', () => {
  it('accepts an ordinary list', () => {
    expect(getWildcardValuesError(['red', 'green'])).toBeNull();
  });

  it('tells a long list apart from a long value', () => {
    expect(getWildcardValuesError(Array.from({ length: 10_001 }, () => 'v'))).toBe('tooManyValues');
    expect(getWildcardValuesError(['ok', 'x'.repeat(2_001)])).toBe('valueTooLong');
  });

  it('accepts the limits exactly', () => {
    expect(getWildcardValuesError(Array.from({ length: 10_000 }, () => 'v'))).toBeNull();
    expect(getWildcardValuesError(['x'.repeat(2_000)])).toBeNull();
  });
});
