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

describe('wildcard file formats', () => {
  it.each([
    ['colours.txt', 'colours'],
    ['animals/dogs.txt', 'animals/dogs'],
    ['./wildcards/moods.TXT', 'wildcards/moods'],
    ['colours', 'colours'],
  ])('derives a wildcard name from %s', (path, expected) => {
    expect(getWildcardNameFromPath(path)).toBe(expected);
  });

  it('normalizes text values without damaging dynamic syntax', () => {
    expect(parseWildcardTextFile('colours.txt', '# list\r\n red \r\n\r\n{green|blue} # choice\n__other__')).toEqual({
      name: 'colours',
      values: ['red', '{green|blue}', '__other__'],
    });
  });

  it('flattens nested records, normalizes values, and ignores non-list leaves', () => {
    expect(
      wildcardsFromNestedRecord({
        animals: { dogs: ['corgi'] },
        'animals/cats': ['tabby'],
        note: 'ignored',
        colours: ['poster #1', '#ff0000', '', 1970, true],
      })
    ).toEqual([
      { name: 'animals/dogs', values: ['corgi'] },
      { name: 'animals/cats', values: ['tabby'] },
      { name: 'colours', values: ['poster', '1970', 'true'] },
    ]);
    expect(wildcardsFromNestedRecord(null)).toEqual([]);
  });

  it('nests slash names and round-trips parent/child collisions independent of order', () => {
    const parent = { name: 'animals', values: ['bear'] };
    const child = { name: 'animals/dogs', values: ['corgi'] };
    const colours = { name: 'colours', values: ['red'] };
    const nested = wildcardsToNestedRecord([child, parent, colours]);

    expect(nested).toEqual({ animals: ['bear'], 'animals/dogs': ['corgi'], colours: ['red'] });
    expect(wildcardsToNestedRecord([parent, child])).toEqual(wildcardsToNestedRecord([child, parent]));
    expect(wildcardsFromNestedRecord(nested).sort((a, b) => a.name.localeCompare(b.name))).toEqual(
      [parent, child, colours].sort((a, b) => a.name.localeCompare(b.name))
    );
  });

  it('does not traverse Object.prototype for user-authored path segments', () => {
    const names = ['constructor', 'toString', 'valueOf', 'hasOwnProperty'];
    const wildcards = names.map((name) => ({ name: `${name}/keys`, values: ['red'] }));
    const nested = wildcardsToNestedRecord(wildcards);

    expect(typeof Object.keys).toBe('function');
    expect(Object.keys(nested)).toEqual(names);
    expect(wildcardsFromNestedRecord(nested)).toEqual(wildcards);
    expect(JSON.stringify(wildcardsToNestedRecord([{ name: 'constructor', values: ['red'] }]))).toBe(
      '{"constructor":["red"]}'
    );
  });
});

describe('wildcard import planning', () => {
  const existing = [{ id: 'w1', name: 'colours' }];

  it('marks catalog clashes and distinguishes rejection causes', () => {
    const entries = planWildcardImport(
      [
        { name: 'colours', values: ['red'] },
        { name: 'new', values: ['calm'] },
        { name: 'bad name!', values: ['x'] },
        { name: 'a'.repeat(129), values: ['x'] },
        { name: 'empty', values: [] },
        { name: 'many', values: Array.from({ length: 10_001 }, (_, index) => String(index)) },
        { name: 'long', values: ['x'.repeat(2_001)] },
      ],
      existing
    );

    expect(entries.map(({ conflictId, rejection }) => ({ conflictId, rejection }))).toEqual([
      { conflictId: 'w1', rejection: null },
      { conflictId: null, rejection: null },
      { conflictId: null, rejection: 'invalid' },
      { conflictId: null, rejection: 'tooLong' },
      { conflictId: null, rejection: 'noValues' },
      { conflictId: null, rejection: 'tooManyValues' },
      { conflictId: null, rejection: 'valueTooLong' },
    ]);
  });

  it('only lets an accepted entry claim a repeated name', () => {
    expect(
      planWildcardImport(
        [
          { name: 'moods', values: [] },
          { name: 'moods', values: ['calm'] },
          { name: 'moods', values: ['tense'] },
        ],
        []
      ).map(({ rejection }) => rejection)
    ).toEqual(['noValues', null, 'duplicate']);
  });

  it('finds a valid available suffix within the name limit', () => {
    expect(getAvailableWildcardName('colours', new Set(['colours', 'colours-2']))).toBe('colours-3');
    expect(getAvailableWildcardName('a'.repeat(128), new Set())).toMatch(/^a+-2$/);
    expect(getAvailableWildcardName(`${'a'.repeat(125)}b--`, new Set())).toMatch(/^a+b-2$/);
  });

  it('creates, replaces, skips, and renames planned entries without collisions', () => {
    const entries = planWildcardImport(
      [
        { name: 'colours', values: ['red'] },
        { name: 'colours-2', values: ['green'] },
        { name: 'moods', values: ['calm'] },
        { name: 'bad name!', values: ['x'] },
      ],
      existing
    );

    expect(getWildcardImportActions(entries, {}, new Set(['colours']))).toEqual([
      { name: 'colours-2', values: ['green'] },
      { name: 'moods', values: ['calm'] },
    ]);
    expect(getWildcardImportActions(entries, { colours: 'replace' }, new Set(['colours']))).toContainEqual({
      id: 'w1',
      name: 'colours',
      values: ['red'],
    });
    expect(getWildcardImportActions(entries, { colours: 'keepBoth' }, new Set(['colours']))).toEqual([
      { name: 'colours-3', values: ['red'] },
      { name: 'colours-2', values: ['green'] },
      { name: 'moods', values: ['calm'] },
    ]);
  });

  it('allocates a distinct suffix to each keep-both clash', () => {
    const entries = planWildcardImport(
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
      getWildcardImportActions(entries, { colours: 'keepBoth', moods: 'keepBoth' }, new Set(['colours', 'moods']))
    ).toEqual([
      { name: 'colours-2', values: ['red'] },
      { name: 'moods-2', values: ['calm'] },
    ]);
  });
});
