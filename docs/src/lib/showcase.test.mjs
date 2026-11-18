import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { sortShowcase } from './showcase.mjs';

/** @param {string} id @param {string} date @param {string} title */
const entry = (id, date, title) => ({ id, data: { date: new Date(date), title } });

describe('sortShowcase', () => {
  it('puts the newest entries first', () => {
    const entries = [entry('a', '2026-01-05', 'A'), entry('b', '2026-03-01', 'B'), entry('c', '2025-12-31', 'C')];
    assert.deepEqual(
      sortShowcase(entries).map(({ id }) => id),
      ['b', 'a', 'c'],
    );
  });

  it('orders entries from the same date by title, then id', () => {
    const entries = [
      entry('z', '2026-02-02', 'Harbour'),
      entry('y', '2026-02-02', 'Atlas'),
      entry('x', '2026-02-02', 'Harbour'),
    ];
    assert.deepEqual(
      sortShowcase(entries).map(({ id }) => id),
      ['y', 'x', 'z'],
    );
  });

  it('leaves the given array as it was', () => {
    const entries = [entry('a', '2026-01-01', 'A'), entry('b', '2026-02-01', 'B')];
    sortShowcase(entries);
    assert.deepEqual(
      entries.map(({ id }) => id),
      ['a', 'b'],
    );
  });
});
