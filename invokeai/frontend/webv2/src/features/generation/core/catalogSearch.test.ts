import { describe, expect, it } from 'vitest';

import { searchCatalog, searchProse } from './catalogSearch';

interface Record {
  name: string;
  body: string[];
}

const record = (name: string, ...body: string[]): Record => ({ body, name });

/** Module-level, as every real caller's accessor is — the cache is keyed by it. */
const getBody = (item: Record): readonly string[] => item.body;

describe('searchCatalog', () => {
  it('returns a copy of everything for an empty query', () => {
    const items = [record('colours', 'red'), record('animals', 'dog')];

    expect(searchCatalog(items, '', getBody)).toEqual(items);
    expect(searchCatalog(items, '   ', getBody)).toEqual(items);
    expect(searchCatalog(items, '', getBody)).not.toBe(items);
  });

  it('matches a name fuzzily', () => {
    const dogs = record('animals/dogs', 'poodle');
    const colours = record('colours', 'red');

    expect(searchCatalog([dogs, colours], 'adg', getBody)).toEqual([dogs]);
  });

  it('ranks name matches above prose matches', () => {
    const byProse = record('animals', 'red panda');
    const byName = record('red', 'crimson');

    expect(searchCatalog([byProse, byName], 'red', getBody)).toEqual([byName, byProse]);
  });

  it('lists a record once when both its name and its prose match', () => {
    const both = record('red', 'red');

    expect(searchCatalog([both], 'red', getBody)).toEqual([both]);
  });

  it('matches prose without regard to case', () => {
    const item = record('colours', 'Crimson', 'CYBERPUNK');

    expect(searchCatalog([item], 'cyber', getBody)).toEqual([item]);
    expect(searchCatalog([item], 'CRIM', getBody)).toEqual([item]);
  });

  it('matches prose by substring, not fuzzily', () => {
    const item = record('colours', 'crimson');

    // Fuzzy would find these; on free text they are noise.
    expect(searchCatalog([item], 'cmn', getBody)).toEqual([]);
    expect(searchCatalog([item], 'crn', getBody)).toEqual([]);
  });

  it('never matches across the boundary between two prose fields', () => {
    const item = record('colours', 'deep red', 'blue steel');

    expect(searchCatalog([item], 'redblue', getBody)).toEqual([]);
    // A space is what a joined haystack would have left matchable here.
    expect(searchCatalog([item], 'red blue', getBody)).toEqual([]);
    expect(searchCatalog([item], 'deep red', getBody)).toEqual([item]);
  });

  it('returns nothing when neither the name nor the prose matches', () => {
    expect(searchCatalog([record('colours', 'red')], 'zzzz', getBody)).toEqual([]);
  });

  it('scans long prose to the end rather than sampling it', () => {
    const long = record('big', ...Array.from({ length: 500 }, (_, index) => `value-${index}`), 'needle');

    expect(searchCatalog([long], 'needle', getBody)).toEqual([long]);
  });

  it('reads an edited record afresh', () => {
    const before = record('colours', 'red');
    const after = { ...before, body: ['blue'] };

    expect(searchCatalog([before], 'blue', getBody)).toEqual([]);
    expect(searchCatalog([after], 'blue', getBody)).toEqual([after]);
  });

  it('keeps two accessors over one record apart', () => {
    const item = { ...record('colours', 'red'), extra: ['moody'] };
    const getExtra = (candidate: typeof item): readonly string[] => candidate.extra;

    expect(searchCatalog([item], 'red', getBody)).toEqual([item]);
    expect(searchCatalog([item], 'red', getExtra)).toEqual([]);
    expect(searchCatalog([item], 'moody', getExtra)).toEqual([item]);
    expect(searchCatalog([item], 'moody', getBody)).toEqual([]);
  });
});

describe('searchProse', () => {
  it('returns a copy of everything for an empty query', () => {
    const items = [record('a', 'red'), record('b', 'blue')];

    expect(searchProse(items, '  ', getBody)).toEqual(items);
    expect(searchProse(items, '  ', getBody)).not.toBe(items);
  });

  it('ignores the name entirely', () => {
    const item = record('crimson', 'red');

    expect(searchProse([item], 'crimson', getBody)).toEqual([]);
    expect(searchProse([item], 'red', getBody)).toEqual([item]);
  });

  it('keeps the given order', () => {
    const first = record('a', 'red one');
    const second = record('b', 'red two');

    expect(searchProse([first, second], 'red', getBody)).toEqual([first, second]);
  });
});
