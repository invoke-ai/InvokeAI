import { describe, expect, it } from 'vitest';

import { createSelectorCache, createStableSelector } from './selectors';

describe('createSelectorCache', () => {
  it('uses shallow equality by default for derived objects', () => {
    const cache = createSelectorCache((snapshot: { count: number; label: string }) => ({ count: snapshot.count }));

    const first = cache.read({ count: 1, label: 'idle' });
    const second = cache.read({ count: 1, label: 'loading' });
    const third = cache.read({ count: 2, label: 'loading' });

    expect(second).toBe(first);
    expect(third).not.toBe(first);
    expect(third).toEqual({ count: 2 });
  });

  it('compares Set contents by default', () => {
    const cache = createSelectorCache((snapshot: { ids: string[] }) => new Set(snapshot.ids));

    const first = cache.read({ ids: ['a', 'b'] });
    const second = cache.read({ ids: ['a', 'b'] });
    const third = cache.read({ ids: ['a', 'c'] });

    expect(second).toBe(first);
    expect(third).not.toBe(first);
  });
});

describe('createStableSelector', () => {
  it('reuses the previous result when derived values are equal', () => {
    const selector = createStableSelector(
      (value: { ids: string[] }) => value.ids.map((id) => id.toUpperCase()),
      (left, right) => left.length === right.length && left.every((id, index) => id === right[index])
    );

    const first = selector({ ids: ['a', 'b'] });
    const second = selector({ ids: ['a', 'b'] });
    const third = selector({ ids: ['a', 'c'] });

    expect(second).toBe(first);
    expect(third).not.toBe(first);
    expect(third).toEqual(['A', 'C']);
  });
});
