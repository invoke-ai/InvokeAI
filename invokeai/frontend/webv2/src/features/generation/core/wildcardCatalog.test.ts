import { describe, expect, it } from 'vitest';

import { filterWildcards, groupWildcardsByPrefix } from './wildcardCatalog';

const wildcard = (name: string, values: string[] = []) => ({ name, values });

describe('filterWildcards', () => {
  it('returns every wildcard for an empty query', () => {
    const wildcards = [wildcard('colours'), wildcard('artists')];

    expect(filterWildcards(wildcards, '')).toEqual(wildcards);
    expect(filterWildcards(wildcards, '   ')).toEqual(wildcards);
  });

  it('returns a copy rather than the caller’s array', () => {
    const wildcards = [wildcard('colours')];

    expect(filterWildcards(wildcards, '')).not.toBe(wildcards);
  });

  it('matches names fuzzily', () => {
    const wildcards = [wildcard('animals/dogs'), wildcard('colours')];

    expect(filterWildcards(wildcards, 'adg')).toEqual([wildcards[0]]);
  });

  it('matches values by substring, case-insensitively', () => {
    const wildcards = [wildcard('moods', ['Cyberpunk', 'serene']), wildcard('colours', ['red'])];

    expect(filterWildcards(wildcards, 'cyberpunk')).toEqual([wildcards[0]]);
    expect(filterWildcards(wildcards, 'CYBER')).toEqual([wildcards[0]]);
  });

  it('ranks name matches above value-only matches', () => {
    const valueOnly = wildcard('moods', ['red mist']);
    const byName = wildcard('red', ['crimson']);

    expect(filterWildcards([valueOnly, byName], 'red')).toEqual([byName, valueOnly]);
  });

  it('lists a wildcard once when both its name and its values match', () => {
    const both = wildcard('red', ['dark red', 'bright red']);

    expect(filterWildcards([both], 'red')).toEqual([both]);
  });

  it('returns nothing when neither names nor values match', () => {
    expect(filterWildcards([wildcard('colours', ['red'])], 'zzzz')).toEqual([]);
  });

  it('finds a match in the last of many values', () => {
    const long = wildcard('big', [...Array.from({ length: 500 }, (_, index) => `value-${index}`), 'needle']);

    expect(filterWildcards([long], 'needle')).toEqual([long]);
  });
});

describe('groupWildcardsByPrefix', () => {
  it('leaves a flat catalog in one unlabelled group', () => {
    const wildcards = [wildcard('colours'), wildcard('artists')];

    expect(groupWildcardsByPrefix(wildcards)).toEqual([{ label: null, wildcards }]);
  });

  it('groups by the segment before the first slash', () => {
    const dogs = wildcard('animals/dogs');
    const cats = wildcard('animals/cats');

    expect(groupWildcardsByPrefix([dogs, cats])).toEqual([{ label: 'animals', wildcards: [dogs, cats] }]);
  });

  it('splits on the first slash only, so deeper names share one header', () => {
    const elves = wildcard('characters/fantasy/elves');
    const orcs = wildcard('characters/scifi/orcs');

    expect(groupWildcardsByPrefix([elves, orcs])).toEqual([{ label: 'characters', wildcards: [elves, orcs] }]);
  });

  it('puts top-level names first, then labelled groups in first-appearance order', () => {
    const dogs = wildcard('animals/dogs');
    const colours = wildcard('colours');
    const warm = wildcard('moods/warm');

    expect(groupWildcardsByPrefix([dogs, colours, warm])).toEqual([
      { label: null, wildcards: [colours] },
      { label: 'animals', wildcards: [dogs] },
      { label: 'moods', wildcards: [warm] },
    ]);
  });

  it('omits the unlabelled group when every name is nested', () => {
    const dogs = wildcard('animals/dogs');

    expect(groupWildcardsByPrefix([dogs])).toEqual([{ label: 'animals', wildcards: [dogs] }]);
  });

  it('preserves the incoming order within a group', () => {
    const cats = wildcard('animals/cats');
    const dogs = wildcard('animals/dogs');

    expect(groupWildcardsByPrefix([dogs, cats])[0]?.wildcards).toEqual([dogs, cats]);
  });

  it('treats a leading slash as ungrouped rather than an empty label', () => {
    const odd = wildcard('/leading');

    expect(groupWildcardsByPrefix([odd])).toEqual([{ label: null, wildcards: [odd] }]);
  });

  it('returns nothing for an empty catalog', () => {
    expect(groupWildcardsByPrefix([])).toEqual([]);
  });
});
