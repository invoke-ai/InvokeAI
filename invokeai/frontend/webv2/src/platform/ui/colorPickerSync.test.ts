import { describe, expect, it } from 'vitest';

import { shouldSyncExternalColor } from './colorPickerSync';

describe('shouldSyncExternalColor', () => {
  it('does not sync when the external value is unchanged', () => {
    expect(shouldSyncExternalColor('#808080', '#808080', '#808080')).toBe(false);
  });

  it('does not sync when the external value changed to exactly what we last emitted (our own round trip)', () => {
    // This is the grey-hue-loss scenario: the user drags the hue slider at
    // S=0, we emit "#808080", the consumer stores it and passes it back as
    // the new `value` prop -- that must not force a re-parse that would
    // collapse the hue we're holding onto internally.
    expect(shouldSyncExternalColor('#808080', '#7f7f7f', '#808080')).toBe(false);
  });

  it('syncs when the external value changed to something other than what we last emitted', () => {
    // A genuine external change (e.g. a programmatic reset, or a different
    // control writing to the same underlying value) should still win.
    expect(shouldSyncExternalColor('#ff0000', '#808080', '#808080')).toBe(true);
  });

  it('syncs on the very first divergence even if nothing has been emitted yet', () => {
    expect(shouldSyncExternalColor('#ff0000', '#000000', '#000000')).toBe(true);
  });

  it('does not sync when the previous and external values are identical, regardless of last emitted', () => {
    expect(shouldSyncExternalColor('#123456', '#123456', '#abcdef')).toBe(false);
  });
});
