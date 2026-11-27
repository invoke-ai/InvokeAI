import { describe, expect, it } from 'vitest';

import { listRowsFromItems, listRowsFromSections } from './listRows';

interface Item {
  id: string;
}

const getId = (item: Item): string => item.id;

describe('listRowsFromSections', () => {
  it('interleaves one header per non-empty section, carrying the item count', () => {
    const rows = listRowsFromSections(
      [
        { items: [{ id: 'a' }, { id: 'b' }], key: 'main', label: 'Main' },
        { items: [], key: 'empty', label: 'Nothing here' },
        { items: [{ id: 'c' }], key: 'lora', label: 'LoRAs' },
      ],
      getId
    );

    expect(rows).toEqual([
      { count: 2, key: 'header:main', kind: 'header', label: 'Main' },
      { item: { id: 'a' }, key: 'a', kind: 'item' },
      { item: { id: 'b' }, key: 'b', kind: 'item' },
      { count: 1, key: 'header:lora', kind: 'header', label: 'LoRAs' },
      { item: { id: 'c' }, key: 'c', kind: 'item' },
    ]);
  });

  it('keeps header keys distinct from item keys that share a section name', () => {
    const rows = listRowsFromSections([{ items: [{ id: 'main' }], key: 'main', label: 'Main' }], getId);

    expect(new Set(rows.map((row) => row.key)).size).toBe(rows.length);
  });
});

describe('listRowsFromItems', () => {
  it('produces item rows only, in order', () => {
    expect(listRowsFromItems([{ id: 'b' }, { id: 'a' }], getId)).toEqual([
      { item: { id: 'b' }, key: 'b', kind: 'item' },
      { item: { id: 'a' }, key: 'a', kind: 'item' },
    ]);
  });
});
