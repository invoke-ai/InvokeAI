import { describe, expect, it } from 'vitest';

import { firstPartyHotkeyCatalog } from './catalog';

describe('firstPartyHotkeyCatalog', () => {
  it('keeps legacy default hotkey parity', () => {
    expect(firstPartyHotkeyCatalog).toHaveLength(91);
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('app.invoke');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('canvas.mergeDown');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('workflows.copySelection');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('gallery.galleryNavLeft');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('gallery.remix');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('viewer.deleteImage');
  });
});
