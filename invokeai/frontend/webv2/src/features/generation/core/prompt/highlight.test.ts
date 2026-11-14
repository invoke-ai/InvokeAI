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

    // Strip comments before wildcard lookup.
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
      // Resolve globs against catalog matches, not literal wildcard names.
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

  describe('for a model that reads the prompt as plain text', () => {
    it('leaves weights, groups, embeddings, functions and bracket errors plain', () => {
      const prompt = "cat+ (dog)1.2 <emb> ('a', 'b').and() \\( unclosed(";

      expect(buildPromptHighlightSegments(prompt, { syntaxProfile: 'none' })).toEqual([
        { kind: 'text', range: { start: 0, end: prompt.length }, text: prompt },
      ]);
    });

    it('still marks dynamic-prompt syntax, which is expanded before any encoder reads the prompt', () => {
      const prompt = '{red+|green} # note';
      const segments = buildPromptHighlightSegments(prompt, { syntaxProfile: 'none', dynamicPrompts: true });

      expect(segments).toEqual([
        { kind: 'variantBrace', range: { start: 0, end: 1 }, text: '{' },
        { kind: 'text', range: { start: 1, end: 5 }, text: 'red+' },
        { kind: 'variantSeparator', range: { start: 5, end: 6 }, text: '|' },
        { kind: 'text', range: { start: 6, end: 11 }, text: 'green' },
        { kind: 'variantBrace', range: { start: 11, end: 12 }, text: '}' },
        { kind: 'text', range: { start: 12, end: 13 }, text: ' ' },
        { kind: 'comment', range: { start: 13, end: 19 }, text: '# note' },
      ]);
    });
  });

  describe('Anima attention-only syntax', () => {
    it.each([
      ['(cat)-1', '-', 'attention'],
      ['(cat)+1', '+', 'attention'],
      ['(cat).5', '.5', 'attentionNumeric'],
      ['(cat)2.', '2.', 'attentionNumeric'],
      ['(cat)2next', '2', 'attentionNumeric'],
      ['(cat)++dog', '++', 'attention'],
    ] as const)('highlights the supported group suffix in %s', (prompt, suffix, kind) => {
      const segments = buildPromptHighlightSegments(prompt, { syntaxProfile: 'anima' });
      expect(segments).toContainEqual(expect.objectContaining({ text: suffix, kind }));
      expect(segments.map((segment) => segment.text).join('')).toBe(prompt);
    });

    it('keeps earlier weights visible when later text has unsupported syntax', () => {
      const prompt = 'foo++bar.baz foo++bar=2 foo"bar+ foo|bar+';
      const segments = buildPromptHighlightSegments(prompt, { syntaxProfile: 'anima' });
      expect(segments.filter((segment) => segment.kind === 'attention').map((segment) => segment.text)).toEqual([
        '++',
        '++',
        '+',
      ]);
      expect(segments.map((segment) => segment.text).join('')).toBe(prompt);
    });

    it('highlights supported word and group weights across Compel-style word boundaries', () => {
      const prompt = 'blue-green+ hair!+ a+b+ foo++bar+ (long hair)2';
      const segments = buildPromptHighlightSegments(prompt, { syntaxProfile: 'anima' });

      expect(segments.filter((segment) => segment.kind === 'attention').map((segment) => segment.text)).toEqual([
        '+',
        '+',
        '+',
        '++',
        '+',
      ]);
      expect(segments.find((segment) => segment.kind === 'attentionNumeric')?.text).toBe('2');
      expect(segments.map((segment) => segment.text).join('')).toBe(prompt);
    });

    it('leaves Compel-only functions, embeddings, and dot/equal suffixes plain', () => {
      const prompt = 'foo.bar+ foo=bar+ <embedding> (red, blue).blend(0.5, 0.5)';
      const segments = buildPromptHighlightSegments(prompt, { syntaxProfile: 'anima' });

      expect(segments.find((segment) => segment.text.includes('foo.bar+'))?.kind).toBe('text');
      expect(segments.find((segment) => segment.text.includes('foo=bar+'))?.kind).toBe('text');
      expect(segments.find((segment) => segment.text.includes('<embedding>'))?.kind).toBe('text');
      expect(segments.find((segment) => segment.text.includes('.blend(0.5, 0.5)'))?.kind).toBe('text');
      expect(segments.some((segment) => segment.kind === 'promptFunctionMethod' || segment.kind === 'embedding')).toBe(
        false
      );
    });

    it('keeps dynamic expansion highlighting while Anima methods are plain', () => {
      const prompt = '{red+|blue} .and()';
      const segments = buildPromptHighlightSegments(prompt, { syntaxProfile: 'anima', dynamicPrompts: true });

      expect(segments.some((segment) => segment.kind === 'variantBrace')).toBe(true);
      expect(segments.find((segment) => segment.text.includes('.and()'))?.kind).toBe('text');
    });

    it('highlights attention in the group before a plain Compel-only method', () => {
      const prompt = 'blue-green+ (red+, blue).and()';
      const segments = buildPromptHighlightSegments(prompt, { syntaxProfile: 'anima' });

      expect(segments.filter((segment) => segment.kind === 'attention').map((segment) => segment.text)).toEqual([
        '+',
        '+',
      ]);
      expect(segments.find((segment) => segment.text.includes('.and()'))?.kind).toBe('text');
      expect(segments.map((segment) => segment.text).join('')).toBe(prompt);
    });
  });
});

// A variant-heavy fixture exposes quadratic annotation lookup.
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

describe('Anima painting follows parsed attention', () => {
  it.each(['red\\+', 'cat)2', '((red)2', '(red (blue)2', 'a(b)2', 'foo.bar+'])(
    'leaves literal suffixes plain in %s',
    (prompt) => {
      const segments = buildPromptHighlightSegments(prompt, { syntaxProfile: 'anima' });
      expect(segments.map(({ text }) => text).join('')).toBe(prompt);
      expect(segments.every(({ kind }) => kind === 'text')).toBe(true);
    }
  );

  it.each([
    '.and(dog+)',
    '.and(dog+',
    '.and(.blend(dog+))',
    '(red, blue).and(dog+)',
    '(red).and((dog)+, cat+)',
    '(cat)2.and(dog+)',
  ])('shows supported attention inside unsupported operations: %s', (prompt) => {
    const segments = buildPromptHighlightSegments(prompt, { syntaxProfile: 'anima' });
    expect(segments.map(({ text }) => text).join('')).toBe(prompt);
    expect(segments.some(({ text, kind }) => text.includes('+') && kind === 'attention')).toBe(true);
    const method = prompt.indexOf('.and');
    expect(segments.find(({ range }) => range.start <= method && range.end > method)?.kind).toBe('text');
  });

  it.each(['red\\.blue+', '"red (blue)2"+', '(foo.bar+)2', 'red+(blue)2', 'red .attend (2)'])(
    'paints attention accepted by the backend in %s',
    (prompt) => {
      const segments = buildPromptHighlightSegments(prompt, { syntaxProfile: 'anima' });
      expect(segments.map(({ text }) => text).join('')).toBe(prompt);
      expect(segments.some(({ kind }) => kind === 'attention' || kind === 'attentionNumeric')).toBe(true);
    }
  );

  it('paints 1500 nested groups without recursion and keeps malformed input plain', () => {
    const valid = `${'('.repeat(1500)}red${')'.repeat(1500)}2`;
    expect(buildPromptHighlightSegments(valid, { syntaxProfile: 'anima' }).at(-1)?.kind).toBe('attentionNumeric');
    const invalid = `${'('.repeat(20_000)}red)2`;
    expect(buildPromptHighlightSegments(invalid, { syntaxProfile: 'anima' })).toEqual([
      { kind: 'text', range: { start: 0, end: invalid.length }, text: invalid },
    ]);
  });
});

it.each(['red .attend (2)', '(red) .attend(++)', '"red".attend(--)'])(
  'keeps the method marker plain and colours its supported weight: %s',
  (prompt) => {
    const segments = buildPromptHighlightSegments(prompt, { syntaxProfile: 'anima' });
    const marker = prompt.indexOf('.attend');
    expect(segments.find(({ range }) => range.start <= marker && range.end > marker)?.kind).toBe('text');
    expect(segments.some(({ kind }) => kind === 'attention' || kind === 'attentionNumeric')).toBe(true);
    expect(segments.map(({ text }) => text).join('')).toBe(prompt);
  }
);

it.each(['("cat(")2', '("cat(")65505', String.raw`"cat\fdog"2`])(
  'keeps rejected quoted attention plain: %s',
  (prompt) => {
    const segments = buildPromptHighlightSegments(prompt, { syntaxProfile: 'anima' });
    expect(segments.map(({ text }) => text).join('')).toBe(prompt);
    expect(segments.every(({ kind }) => kind === 'text')).toBe(true);
  }
);

it.each(['n', 't', 'r'])('paints source ranges after a quoted \\%s separator', (escape) => {
  const prompt = `"cat\\${escape}dog+"2`;
  const plus = prompt.indexOf('+');
  const segments = buildPromptHighlightSegments(prompt, { syntaxProfile: 'anima' });
  expect(segments.map(({ text }) => text).join('')).toBe(prompt);
  expect(segments.find(({ range }) => range.start <= plus && range.end > plus)?.kind).toBe('attention');
  expect(segments.find(({ range }) => range.start <= 4 && range.end > 4)?.kind).toBe('text');
});

it.each([
  String.raw`"cat\fdog+"2 bird+`,
  '"cat("2 bird+',
  '"cat)"2 bird+',
  String.raw`"cat\fdog+"2,bird+`,
  String.raw`.and("cat\fdog+"2, bird+)`,
])('paints only the valid following weight after a closed rejected quote: %s', (prompt) => {
  const segments = buildPromptHighlightSegments(prompt, { syntaxProfile: 'anima' });
  expect(segments.map(({ text }) => text).join('')).toBe(prompt);
  const bird = prompt.indexOf('bird');
  expect(
    segments.filter(({ kind }) => kind === 'attention' || kind === 'attentionNumeric').map(({ text }) => text)
  ).toEqual(['+']);
  expect(segments.find(({ range }) => range.start <= bird + 4 && range.end > bird + 4)?.kind).toBe('attention');
});
