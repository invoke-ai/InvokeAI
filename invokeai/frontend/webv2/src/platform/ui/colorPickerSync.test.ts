import { describe, expect, it } from 'vitest';

import { shouldSyncExternalColor } from './colorPickerSync';

describe('shouldSyncExternalColor', () => {
  it('does not sync when the external value is unchanged', () => {
    expect(shouldSyncExternalColor('#808080', '#808080', '#808080', true)).toBe(false);
    expect(shouldSyncExternalColor('#808080', '#808080', '#808080', false)).toBe(false);
  });

  it('does not sync when the external value changed to exactly what we last emitted (our own round trip)', () => {
    // Echoing grey hex must not discard the picker's hidden hue.
    expect(shouldSyncExternalColor('#808080', '#7f7f7f', '#808080', true)).toBe(false);
  });

  it('syncs an external change back to the last-emitted value once its echo was consumed', () => {
    // After the echo settles, returning to the last emitted value is an independent change.
    expect(shouldSyncExternalColor('#808080', '#7f7f7f', '#808080', false)).toBe(true);
  });

  it('syncs when the external value changed to something other than what we last emitted', () => {
    expect(shouldSyncExternalColor('#ff0000', '#808080', '#808080', true)).toBe(true);
  });

  it('syncs on the very first divergence even if nothing has been emitted yet', () => {
    expect(shouldSyncExternalColor('#ff0000', '#000000', '#000000', false)).toBe(true);
  });

  it('does not sync when the previous and external values are identical, regardless of last emitted', () => {
    expect(shouldSyncExternalColor('#123456', '#123456', '#abcdef', true)).toBe(false);
  });

  it('treats a differently-cased echo of what we emitted as our own round trip', () => {
    expect(shouldSyncExternalColor('#FF0000', '#808080', '#ff0000', true)).toBe(false);
  });

  it('treats a shorthand echo of what we emitted as our own round trip', () => {
    expect(shouldSyncExternalColor('#f00', '#808080', '#ff0000', true)).toBe(false);
  });

  it('treats an opaque 8-digit echo of a 6-digit emit as our own round trip', () => {
    expect(shouldSyncExternalColor('#ff0000ff', '#808080', '#ff0000', true)).toBe(false);
  });

  it('syncs when only the alpha differs', () => {
    expect(shouldSyncExternalColor('#ff000080', '#ff0000ff', '#ff0000ff', true)).toBe(true);
  });

  it('round-trips 8-digit values without re-syncing', () => {
    expect(shouldSyncExternalColor('#ff000080', '#ff0000ff', '#FF000080', true)).toBe(false);
  });

  it('passes non-hex values through without treating them as equal', () => {
    // Unparseable colors compare verbatim, not through a shared sentinel.
    expect(shouldSyncExternalColor('transparent', 'currentColor', 'currentColor', true)).toBe(true);
    expect(shouldSyncExternalColor('transparent', 'currentColor', 'transparent', true)).toBe(false);
  });
});
