import { describe, expect, it } from 'vitest';

import { getNextIconId } from './layoutPresetDialogModel';

describe('layout preset icon keyboard navigation', () => {
  const iconIds = ['grid', 'text', 'image'];

  it.each([
    ['grid', 'ArrowRight', 'text'],
    ['text', 'ArrowDown', 'image'],
    ['image', 'ArrowRight', 'grid'],
    ['grid', 'ArrowLeft', 'image'],
    ['text', 'Home', 'grid'],
    ['text', 'End', 'image'],
  ])('moves from %s with %s to %s', (current, key, expected) => {
    expect(getNextIconId(iconIds, current, key)).toBe(expected);
  });

  it('ignores unrelated keys', () => {
    expect(getNextIconId(iconIds, 'text', 'Enter')).toBeNull();
  });
});
