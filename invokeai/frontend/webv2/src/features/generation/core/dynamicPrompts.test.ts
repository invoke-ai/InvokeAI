import { describe, expect, it } from 'vitest';

import {
  expandsDynamicPrompts,
  findDynamicPromptComment,
  getWildcardNameError,
  getWildcardValuesError,
  hasDynamicPromptSyntax,
  matchesKnownWildcard,
  normalizeWildcardValues,
  planGenerateSubmission,
  scanWildcardReferences,
  sanitizeDynamicPromptsConfig,
  sanitizeMaxPrompts,
  type GenerateSubmissionSource,
} from './dynamicPrompts';

describe('hasDynamicPromptSyntax', () => {
  it('detects a variant anywhere in the prompt', () => {
    expect(hasDynamicPromptSyntax('a {red|green} ball')).toBe(true);
    expect(hasDynamicPromptSyntax('{a}')).toBe(true);
    expect(hasDynamicPromptSyntax('multi\nline {a|b}')).toBe(true);
  });

  it('never treats a JSON prompt as dynamic, fenced or not', () => {
    const draft =
      '{"width":1024,"height":1536,"bbox_coord_space":"abs","bbox_format":"xyxy","summary":{"caption":"a #1 poster"},"items":[{"caption":"__colors__"}]}';
    expect(hasDynamicPromptSyntax(draft)).toBe(false);
    expect(hasDynamicPromptSyntax(`\`\`\`json\n${draft}\n\`\`\``)).toBe(false);
    expect(hasDynamicPromptSyntax('[{"caption": "a {red|green} ball"}]')).toBe(false);
    // Braces that do not form JSON are still variants.
    expect(hasDynamicPromptSyntax('{"caption": a {red|green} ball}')).toBe(true);
    expect(hasDynamicPromptSyntax('a poster {"caption": "x"}')).toBe(true);
  });

  it('detects a wildcard reference', () => {
    expect(hasDynamicPromptSyntax('a __colors__ ball')).toBe(true);
    expect(hasDynamicPromptSyntax('prefix__name__')).toBe(true);
    expect(hasDynamicPromptSyntax('a __animals/dogs__ ball')).toBe(true);
    expect(hasDynamicPromptSyntax('a __artists/*__ ball')).toBe(true);
    expect(hasDynamicPromptSyntax('a __~colors__ ball')).toBe(true);
    expect(hasDynamicPromptSyntax('a __outfit(mood=warm)__ ball')).toBe(true);
    expect(hasDynamicPromptSyntax('a __colou?__ __animals/[dc]ogs__ ball')).toBe(true);
  });

  // # requires backend expansion even without braces.
  it('detects a comment', () => {
    expect(hasDynamicPromptSyntax('a red ball # for now')).toBe(true);
    expect(hasDynamicPromptSyntax('a red ball \\# for now')).toBe(true);
  });

  it('ignores prompts with neither a variant nor a wildcard', () => {
    expect(hasDynamicPromptSyntax('a red ball')).toBe(false);
    expect(hasDynamicPromptSyntax('unclosed { brace')).toBe(false);
    // Disallow leading/trailing underscores in wildcard names so delimiters round-trip.
    expect(hasDynamicPromptSyntax('snake__case word')).toBe(false);
  });
});

describe('scanWildcardReferences', () => {
  it('recognizes references adjacent to a word and preserves their exact ranges', () => {
    expect(scanWildcardReferences('prefix__name__')).toEqual([{ lookupPath: 'name', range: { end: 14, start: 6 } }]);
    expect(scanWildcardReferences('snake__case__ word')).toEqual([
      { lookupPath: 'case', range: { end: 13, start: 5 } },
    ]);
  });

  it('returns the full range and lookup path for sampler overrides and parameters', () => {
    const prompt = 'a __~outfit(mood=warm)__ beside __animals/[dc]ogs__';

    expect(scanWildcardReferences(prompt)).toEqual([
      {
        lookupPath: 'outfit',
        range: { end: 24, start: 2 },
      },
      {
        lookupPath: 'animals/[dc]ogs',
        range: { end: 51, start: 32 },
      },
    ]);
  });

  it('does not mistake delimiters inside an ordinary word for a reference', () => {
    expect(scanWildcardReferences('snake__case and __valid__')).toEqual([
      { lookupPath: 'valid', range: { end: 25, start: 16 } },
    ]);
  });

  it('leaves incomplete, malformed, and overlength tokens alone', () => {
    expect(scanWildcardReferences('snake__case')).toEqual([]);
    expect(scanWildcardReferences('__not a name__')).toEqual([]);
    expect(scanWildcardReferences(`__${'x'.repeat(129)}__`)).toEqual([]);
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

describe('expandsDynamicPrompts', () => {
  it('expands dynamic syntax only while dynamic prompts are on', () => {
    expect(expandsDynamicPrompts('a {red|green} ball', true)).toBe(true);
    expect(expandsDynamicPrompts('a {red|green} ball', false)).toBe(false);
    expect(expandsDynamicPrompts('wall painted #ff0000', false)).toBe(false);
    expect(expandsDynamicPrompts('a red ball', true)).toBe(false);
  });
});

describe('findDynamicPromptComment', () => {
  it('finds a # anywhere, since # alone makes the prompt expand', () => {
    expect(findDynamicPromptComment('wall painted #ff0000')).toBe('#');
    expect(findDynamicPromptComment('first line\nsecond # note')).toBe('#');
  });

  it('finds // and block comments only in a prompt expanded for other syntax', () => {
    expect(findDynamicPromptComment('see https://example.com')).toBeNull();
    expect(findDynamicPromptComment('a {red|green} ball, see https://example.com')).toBe('//');
    expect(findDynamicPromptComment('a /* draft */ {red|green} ball')).toBe('/*');
    expect(findDynamicPromptComment('a /* unclosed {red|green} ball')).toBeNull();
  });

  it('names the comment that comes first', () => {
    expect(findDynamicPromptComment('a {b|c} // later # last')).toBe('//');
  });

  it('finds nothing in a prompt without comments', () => {
    expect(findDynamicPromptComment('a {red|green} ball')).toBeNull();
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
      enabled: true,
      maxPrompts: 100,
      resample: true,
      sampleSeed: 0,
      seedBehaviour: 'per-iteration',
    });
    expect(
      sanitizeDynamicPromptsConfig({
        combinatorial: false,
        enabled: false,
        maxPrompts: 7,
        resample: false,
        sampleSeed: 42,
        seedBehaviour: 'per-image',
      })
    ).toEqual({
      combinatorial: false,
      enabled: false,
      maxPrompts: 7,
      resample: false,
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

  it('supports question marks, classes, ranges, and negated classes', () => {
    expect(matchesKnownWildcard('colour?', new Set(['colours']))).toBe(true);
    expect(matchesKnownWildcard('animals/[dc]ogs', CATALOG)).toBe(true);
    expect(matchesKnownWildcard('animals/[a-z]ogs', CATALOG)).toBe(true);
    expect(matchesKnownWildcard('animals/[!c]ogs', CATALOG)).toBe(true);
    expect(matchesKnownWildcard('animals/[!d]ogs', CATALOG)).toBe(false);
  });

  it('treats an unclosed or invalid class like Python fnmatch', () => {
    expect(matchesKnownWildcard('[abc', new Set(['[abc']))).toBe(true);
    expect(matchesKnownWildcard('[abc', new Set(['a']))).toBe(false);
    expect(matchesKnownWildcard('[z-a]', new Set(['z', '-', 'a']))).toBe(false);
    expect(matchesKnownWildcard('[!z-a]', new Set(['q']))).toBe(true);
  });

  it('anchors both ends, so the two cannot share characters', () => {
    // Prefix and suffix cannot overlap: ab*ba requires at least four characters.
    expect(matchesKnownWildcard('ab*ba', new Set(['aba']))).toBe(false);
    expect(matchesKnownWildcard('ab*ba', new Set(['abba']))).toBe(true);
  });

  it('takes the interior runs in order', () => {
    expect(matchesKnownWildcard('a*b*c', new Set(['axbxc']))).toBe(true);
    expect(matchesKnownWildcard('a*b*c', new Set(['axcxb']))).toBe(false);
    // Adjacent stars collapse: `**` means what `*` means.
    expect(matchesKnownWildcard('a**c', new Set(['abc']))).toBe(true);
  });

  // Avoid regex backtracking on user-entered repeated stars.
  it('does not backtrack on a path full of stars', () => {
    const names = new Set(Array.from({ length: 50 }, (_, index) => `colours/warm-autumn-palette-${index}`));
    const started = performance.now();

    expect(matchesKnownWildcard(`${'*'.repeat(24)}q`, names)).toBe(false);
    expect(performance.now() - started).toBeLessThan(100);
  });
});

describe('normalizeWildcardValues', () => {
  // Normalize ingress so storage, export, and reimport agree.
  it('drops blank lines and trims the rest', () => {
    expect(normalizeWildcardValues(['red', '', '  green  ', '   ', 'blue'])).toEqual(['red', 'green', 'blue']);
  });

  it('leaves an already-clean list alone', () => {
    expect(normalizeWildcardValues(['red', 'green'])).toEqual(['red', 'green']);
  });

  // # starts an unescapable comment, including inside a value such as red#blue.
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

const source = (overrides: Partial<GenerateSubmissionSource> = {}): GenerateSubmissionSource => ({
  batchCount: 2,
  dynamicPromptsEnabled: true,
  negativePrompt: 'blurry',
  negativePromptEnabled: true,
  positivePrompt: 'a cat',
  promptTemplate: null,
  seed: 10,
  seedMode: 'increment',
  ...overrides,
});

describe('planGenerateSubmission', () => {
  it('merges the template and suppresses its negative side with the negative prompt', () => {
    const promptTemplate = {
      id: 't',
      name: 'Photo',
      negativePrompt: '{prompt}, lowres',
      positivePrompt: 'photo of {prompt}',
    };

    expect(planGenerateSubmission(source({ promptTemplate })).submission).toMatchObject({
      negativePrompt: 'blurry, lowres',
      positivePrompt: 'photo of a cat',
    });

    const suppressed = planGenerateSubmission(source({ negativePromptEnabled: false, promptTemplate }));

    expect(suppressed.submission.negativePrompt).toBe('');
    expect(suppressed.effectivePrompts.negativePrompt).toBe('blurry, lowres');
  });

  it('sends a JSON prompt whole and unexpanded under a template, with the negative side templated', () => {
    const layout =
      '{"bbox_coord_space":"abs_centralized","bbox_format":"yxyx","summary":{"caption":"a {red|green} cup"},"items":[]}';
    const promptTemplate = {
      id: 't',
      name: 'Photo',
      negativePrompt: '{prompt}, lowres',
      positivePrompt: 'photo of {prompt}',
    };
    const plan = planGenerateSubmission(
      source({ dynamicPromptsSeedBehaviour: 'per-image', positivePrompt: layout, promptTemplate }),
      ['stale expansion']
    );

    expect(plan.submission).toEqual({
      batchCount: 2,
      negativePrompt: 'blurry, lowres',
      positivePrompt: layout,
      seed: 10,
      seedStep: plan.seedPlan.step,
    });
  });

  it('sends an expansion and its seed behaviour only for a prompt with dynamic syntax', () => {
    const expansion = ['a red cat', 'a green cat'];
    const expanded = planGenerateSubmission(
      source({ dynamicPromptsSeedBehaviour: 'per-image', positivePrompt: 'a {red|green} cat' }),
      expansion
    );
    const plain = planGenerateSubmission(source({ dynamicPromptsSeedBehaviour: 'per-image' }), expansion);

    expect(expanded.submission).toEqual({
      batchCount: 2,
      negativePrompt: 'blurry',
      positivePrompt: 'a {red|green} cat',
      positivePrompts: expansion,
      seed: 10,
      seedBehaviour: 'per-image',
      seedStep: 1,
    });
    // Two prompts per image across two runs consume four seeds.
    expect(expanded.seedPlan.nextSeed).toBe(14);
    expect(plain.submission).toEqual({
      batchCount: 2,
      negativePrompt: 'blurry',
      positivePrompt: 'a cat',
      seed: 10,
      seedStep: 1,
    });
    expect(plain.seedPlan.nextSeed).toBe(12);
  });

  it('sends a prompt with dynamic syntax as typed when dynamic prompts are off', () => {
    const { seedPlan, submission } = planGenerateSubmission(
      source({
        dynamicPromptsEnabled: false,
        dynamicPromptsSeedBehaviour: 'per-image',
        positivePrompt: 'a {red|green} cat',
      }),
      ['a red cat', 'a green cat']
    );

    expect(submission).toEqual({
      batchCount: 2,
      negativePrompt: 'blurry',
      positivePrompt: 'a {red|green} cat',
      seed: 10,
      seedStep: 1,
    });
    expect(seedPlan.nextSeed).toBe(12);
  });

  it('holds a fixed seed', () => {
    const { seedPlan, submission } = planGenerateSubmission(source({ seedMode: 'fixed' }));

    expect([submission.seed, submission.seedStep, seedPlan.nextSeed]).toEqual([10, 0, null]);
  });
});
