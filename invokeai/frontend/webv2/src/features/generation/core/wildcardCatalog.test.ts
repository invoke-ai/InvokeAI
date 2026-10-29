import { describe, expect, it } from 'vitest';

import { filterWildcards, groupWildcardsByPrefix } from './wildcardCatalog';

const wildcards = [
  { name: 'animals/dogs', values: ['terrier', 'retriever'] },
  { name: 'colours', values: ['red', 'cyberpunk blue'] },
  { name: 'architecture/gothic', values: ['cathedral'] },
];

describe('filterWildcards', () => {
  it('adapts wildcard names and values to catalog search', () => {
    expect(filterWildcards(wildcards, 'adg')).toEqual([wildcards[0]]);
    expect(filterWildcards(wildcards, 'CYBERPUNK')).toEqual([wildcards[1]]);
    expect(filterWildcards(wildcards, 'missing')).toEqual([]);
  });

  it('returns a copy for an empty query and ranks a name hit above a value-only hit', () => {
    const all = filterWildcards(wildcards, '');
    const ranked = filterWildcards(
      [
        { name: 'first', values: ['colour study'] },
        { name: 'colours', values: ['red'] },
      ],
      'colour'
    );

    expect(all).toEqual(wildcards);
    expect(all).not.toBe(wildcards);
    expect(ranked.map(({ name }) => name)).toEqual(['colours', 'first']);
  });
});

describe('groupWildcardsByPrefix', () => {
  it('groups on the first slash while preserving top-level and incoming order', () => {
    expect(
      groupWildcardsByPrefix([
        { name: 'animals/dogs' },
        { name: 'plain' },
        { name: 'animals/fantasy/elves' },
        { name: 'art/styles' },
      ])
    ).toEqual([
      { label: null, wildcards: [{ name: 'plain' }] },
      { label: 'animals', wildcards: [{ name: 'animals/dogs' }, { name: 'animals/fantasy/elves' }] },
      { label: 'art', wildcards: [{ name: 'art/styles' }] },
    ]);
  });

  it('handles flat, all-nested, leading-slash, and empty catalogs', () => {
    expect(groupWildcardsByPrefix([{ name: 'plain' }])).toEqual([{ label: null, wildcards: [{ name: 'plain' }] }]);
    expect(groupWildcardsByPrefix([{ name: 'animals/dogs' }])).toEqual([
      { label: 'animals', wildcards: [{ name: 'animals/dogs' }] },
    ]);
    expect(groupWildcardsByPrefix([{ name: '/odd' }])).toEqual([{ label: null, wildcards: [{ name: '/odd' }] }]);
    expect(groupWildcardsByPrefix([])).toEqual([]);
  });
});
