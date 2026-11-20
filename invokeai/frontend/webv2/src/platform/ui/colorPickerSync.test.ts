import { describe, expect, it } from 'vitest';

import { shouldSyncExternalColor } from './colorPickerSync';

describe('shouldSyncExternalColor', () => {
  // Columns: external, previous, last emitted, echo pending.
  it.each<[string, string, string, string, boolean, boolean]>([
    ['skips an unchanged external value with an echo pending', '#808080', '#808080', '#808080', true, false],
    ['skips an unchanged external value with no echo pending', '#808080', '#808080', '#808080', false, false],
    // Echoing grey hex must not discard the picker's hidden hue.
    ['skips the echo of exactly what was last emitted', '#808080', '#7f7f7f', '#808080', true, false],
    // After the echo settles, returning to the last emitted value is an independent change.
    [
      'syncs a return to the last-emitted value once its echo was consumed',
      '#808080',
      '#7f7f7f',
      '#808080',
      false,
      true,
    ],
    ['syncs a change to something other than what was last emitted', '#ff0000', '#808080', '#808080', true, true],
    ['syncs the very first divergence before anything was emitted', '#ff0000', '#000000', '#000000', false, true],
    [
      'skips identical previous and external values whatever was last emitted',
      '#123456',
      '#123456',
      '#abcdef',
      true,
      false,
    ],
    ['skips a differently-cased echo of what was emitted', '#FF0000', '#808080', '#ff0000', true, false],
    ['skips a shorthand echo of what was emitted', '#f00', '#808080', '#ff0000', true, false],
    ['skips an opaque 8-digit echo of a 6-digit emit', '#ff0000ff', '#808080', '#ff0000', true, false],
    ['syncs a change of alpha alone', '#ff000080', '#ff0000ff', '#ff0000ff', true, true],
    ['skips the echo of an 8-digit emit', '#ff000080', '#ff0000ff', '#FF000080', true, false],
    // Unparseable colors compare verbatim, not through a shared sentinel.
    ['syncs a non-hex value that differs verbatim', 'transparent', 'currentColor', 'currentColor', true, true],
    ['skips a non-hex echo that matches verbatim', 'transparent', 'currentColor', 'transparent', true, false],
  ])('%s', (_label, external, previous, lastEmitted, echoPending, expected) => {
    expect(shouldSyncExternalColor(external, previous, lastEmitted, echoPending)).toBe(expected);
  });
});
