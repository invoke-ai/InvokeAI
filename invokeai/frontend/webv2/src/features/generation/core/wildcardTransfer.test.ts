import { describe, expect, it } from 'vitest';

import {
  getAvailableWildcardName,
  getWildcardImportActions,
  getWildcardNameFromPath,
  parseWildcardTextFile,
  planWildcardImport,
  wildcardsFromNestedRecord,
  wildcardsToNestedRecord,
} from './wildcardTransfer';

describe('getWildcardNameFromPath', () => {
  it('drops the extension and keeps the nesting', () => {
    expect(getWildcardNameFromPath('colours.txt')).toBe('colours');
    expect(getWildcardNameFromPath('animals/dogs.txt')).toBe('animals/dogs');
    expect(getWildcardNameFromPath('./wildcards/moods.TXT')).toBe('wildcards/moods');
  });

  it('leaves an extensionless path alone', () => {
    expect(getWildcardNameFromPath('colours')).toBe('colours');
  });
});

describe('parseWildcardTextFile', () => {
  it('reads one value per line', () => {
    expect(parseWildcardTextFile('colours.txt', 'red\ngreen\nblue')).toEqual({
      name: 'colours',
      values: ['red', 'green', 'blue'],
    });
  });

  it('drops comments and blank lines', () => {
    const parsed = parseWildcardTextFile('colours.txt', '# a list\nred\n\n  green  # the good one\n');

    expect(parsed.values).toEqual(['red', 'green']);
  });

  it('reads CRLF files', () => {
    expect(parseWildcardTextFile('colours.txt', 'red\r\ngreen').values).toEqual(['red', 'green']);
  });

  it('keeps dynamic syntax inside a value', () => {
    expect(parseWildcardTextFile('c.txt', '{red|green} ball\n__other__').values).toEqual([
      '{red|green} ball',
      '__other__',
    ]);
  });
});

describe('wildcardsFromNestedRecord', () => {
  it('flattens nesting into slash-separated names', () => {
    expect(wildcardsFromNestedRecord({ animals: { cats: ['tabby'], dogs: ['corgi', 'husky'] } })).toEqual([
      { name: 'animals/cats', values: ['tabby'] },
      { name: 'animals/dogs', values: ['corgi', 'husky'] },
    ]);
  });

  it('takes a key that already contains a slash as written', () => {
    expect(wildcardsFromNestedRecord({ 'animals/dogs': ['corgi'] })).toEqual([
      { name: 'animals/dogs', values: ['corgi'] },
    ]);
  });

  it('stringifies scalars an unquoted file produces', () => {
    expect(wildcardsFromNestedRecord({ years: [1970, 3.5, true] })).toEqual([
      { name: 'years', values: ['1970', '3.5', 'true'] },
    ]);
  });

  it('ignores leaves that are not lists, per the spec', () => {
    expect(wildcardsFromNestedRecord({ colours: ['red'], note: 'not a list', nothing: null })).toEqual([
      { name: 'colours', values: ['red'] },
    ]);
  });

  it('drops blank entries', () => {
    expect(wildcardsFromNestedRecord({ colours: ['red', '', '   '] })).toEqual([{ name: 'colours', values: ['red'] }]);
  });

  it('returns nothing for input that is not a mapping', () => {
    expect(wildcardsFromNestedRecord(['red'])).toEqual([]);
    expect(wildcardsFromNestedRecord(null)).toEqual([]);
    expect(wildcardsFromNestedRecord('red')).toEqual([]);
  });
});

describe('wildcardsToNestedRecord', () => {
  it('nests on the slashes', () => {
    expect(
      wildcardsToNestedRecord([
        { name: 'animals/dogs', values: ['corgi'] },
        { name: 'animals/cats', values: ['tabby'] },
        { name: 'colours', values: ['red'] },
      ])
    ).toEqual({ animals: { cats: ['tabby'], dogs: ['corgi'] }, colours: ['red'] });
  });

  // `animals` holds a value list, so `animals/dogs` has nowhere to nest into.
  it('falls back to a flat key when a prefix is itself a wildcard', () => {
    expect(
      wildcardsToNestedRecord([
        { name: 'animals', values: ['bear'] },
        { name: 'animals/dogs', values: ['corgi'] },
      ])
    ).toEqual({ animals: ['bear'], 'animals/dogs': ['corgi'] });
  });

  // Regression: deciding this as the loop went meant the child overwrote the
  // parent, or the parent the child, depending on which came first.
  it('resolves a parent/child clash the same way whichever order they arrive in', () => {
    const parent = { name: 'animals', values: ['bear'] };
    const child = { name: 'animals/dogs', values: ['corgi'] };

    expect(wildcardsToNestedRecord([child, parent])).toEqual(wildcardsToNestedRecord([parent, child]));
  });

  it('round-trips through the nested form', () => {
    const wildcards = [
      { name: 'animals/dogs', values: ['corgi'] },
      { name: 'animals', values: ['bear'] },
      { name: 'colours', values: ['red', 'green'] },
    ];

    expect(
      wildcardsFromNestedRecord(wildcardsToNestedRecord(wildcards)).sort((a, b) => a.name.localeCompare(b.name))
    ).toEqual(wildcards.sort((a, b) => a.name.localeCompare(b.name)));
  });
});

describe('planWildcardImport', () => {
  const existing = [{ id: 'w1', name: 'colours' }];

  it('marks a clash with the wildcard it would overwrite', () => {
    const [clashing, fresh] = planWildcardImport(
      [
        { name: 'colours', values: ['red'] },
        { name: 'moods', values: ['calm'] },
      ],
      existing
    );

    expect(clashing).toMatchObject({ conflictId: 'w1', rejection: null });
    expect(fresh).toMatchObject({ conflictId: null, rejection: null });
  });

  it('reports a name the backend would refuse rather than mangling it', () => {
    const [entry] = planWildcardImport([{ name: 'my colours!', values: ['red'] }], existing);

    expect(entry.rejection).toBe('invalid');
  });

  it('reports an over-long name', () => {
    const [entry] = planWildcardImport([{ name: 'a'.repeat(129), values: ['red'] }], existing);

    expect(entry.rejection).toBe('tooLong');
  });

  it('reports an empty list instead of creating a wildcard that expands to nothing', () => {
    const [entry] = planWildcardImport([{ name: 'moods', values: [] }], existing);

    expect(entry.rejection).toBe('noValues');
  });

  it('reports a list past the backend limits', () => {
    const [tooMany, tooLong] = planWildcardImport(
      [
        { name: 'many', values: Array.from({ length: 10_001 }, (_, index) => `v${index}`) },
        { name: 'long', values: ['x'.repeat(2_001)] },
      ],
      existing
    );

    expect(tooMany.rejection).toBe('tooManyValues');
    expect(tooLong.rejection).toBe('tooManyValues');
  });

  it('keeps the first of a name repeated within one import and reports the rest', () => {
    const entries = planWildcardImport(
      [
        { name: 'moods', values: ['calm'] },
        { name: 'moods', values: ['tense'] },
      ],
      existing
    );

    expect(entries.map((entry) => entry.rejection)).toEqual([null, 'duplicate']);
  });
});

describe('getAvailableWildcardName', () => {
  it('suffixes until the name is free', () => {
    expect(getAvailableWildcardName('colours', new Set(['colours']))).toBe('colours-2');
    expect(getAvailableWildcardName('colours', new Set(['colours', 'colours-2']))).toBe('colours-3');
  });

  it('trims the base so the suffix fits the length limit', () => {
    const name = getAvailableWildcardName('a'.repeat(128), new Set());

    expect(name).toHaveLength(128);
    expect(name.endsWith('-2')).toBe(true);
  });

  // A name may not end in a hyphen, so trimming must not leave one exposed.
  it('leaves a valid name after trimming', () => {
    expect(getAvailableWildcardName(`${'a'.repeat(125)}b--`, new Set())).toMatch(/^a+b-2$/);
  });
});

describe('getWildcardImportActions', () => {
  const entries = planWildcardImport(
    [
      { name: 'colours', values: ['red'] },
      { name: 'moods', values: ['calm'] },
      { name: 'bad name!', values: ['x'] },
    ],
    [{ id: 'w1', name: 'colours' }]
  );
  const existingNames = new Set(['colours']);

  it('creates the new ones and skips clashes by default', () => {
    expect(getWildcardImportActions(entries, {}, existingNames)).toEqual([{ name: 'moods', values: ['calm'] }]);
  });

  it('updates in place when a clash is replaced', () => {
    expect(getWildcardImportActions(entries, { colours: 'replace' }, existingNames)).toEqual([
      { id: 'w1', name: 'colours', values: ['red'] },
      { name: 'moods', values: ['calm'] },
    ]);
  });

  it('renames when a clash keeps both', () => {
    expect(getWildcardImportActions(entries, { colours: 'keepBoth' }, existingNames)).toEqual([
      { name: 'colours-2', values: ['red'] },
      { name: 'moods', values: ['calm'] },
    ]);
  });

  it('never emits an action for a rejected entry', () => {
    const actions = getWildcardImportActions(entries, { 'bad name!': 'replace', colours: 'replace' }, existingNames);

    expect(actions.some((action) => action.name === 'bad name!')).toBe(false);
  });

  // Two "keep both" clashes must not both land on `-2`.
  it('gives each kept-both clash its own name', () => {
    const clashes = planWildcardImport(
      [
        { name: 'colours', values: ['red'] },
        { name: 'moods', values: ['calm'] },
      ],
      [
        { id: 'w1', name: 'colours' },
        { id: 'w2', name: 'moods' },
      ]
    );

    expect(
      getWildcardImportActions(clashes, { colours: 'keepBoth', moods: 'keepBoth' }, new Set(['colours', 'moods']))
    ).toEqual([
      { name: 'colours-2', values: ['red'] },
      { name: 'moods-2', values: ['calm'] },
    ]);
  });
});
