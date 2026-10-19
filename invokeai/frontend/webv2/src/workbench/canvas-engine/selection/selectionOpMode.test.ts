import { describe, expect, it } from 'vitest';

import { selectionOpFor } from './selectionOpMode';

const none = { alt: false, ctrl: false, meta: false, shift: false };

describe('selectionOpFor', () => {
  it('falls back to the persistent mode when no modifier is held', () => {
    expect(selectionOpFor(none, 'replace')).toBe('replace');
    expect(selectionOpFor(none, 'add')).toBe('add');
    expect(selectionOpFor(none, 'subtract')).toBe('subtract');
    expect(selectionOpFor(none, 'intersect')).toBe('intersect');
  });

  it('lets held modifiers override the persistent mode', () => {
    expect(selectionOpFor({ ...none, shift: true }, 'replace')).toBe('add');
    expect(selectionOpFor({ ...none, alt: true }, 'replace')).toBe('subtract');
    expect(selectionOpFor({ ...none, alt: true, shift: true }, 'replace')).toBe('intersect');
  });

  it('overrides a non-replace mode too, so the modifier always wins', () => {
    expect(selectionOpFor({ ...none, shift: true }, 'subtract')).toBe('add');
    expect(selectionOpFor({ ...none, alt: true }, 'add')).toBe('subtract');
  });
});
