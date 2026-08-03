import { describe, expect, it } from 'vitest';

import { firstPartyHotkeyCatalog, OPEN_COMMAND_PALETTE_HOTKEY } from './catalog';

describe('firstPartyHotkeyCatalog', () => {
  it('keeps legacy default hotkey parity', () => {
    // 91 legacy-parity entries + `canvas.newSession` (webv2 new-canvas command,
    // no default keys — Task 46) + `app.openCommandPalette` (webv2-only, mod+k)
    // + the six top-bar redesign commands (alt+mod+enter, mod+p, an unbound save,
    // and alt+1..3).
    expect(firstPartyHotkeyCatalog).toHaveLength(99);
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('app.invoke');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('app.openCommandPalette');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('canvas.mergeDown');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('canvas.newSession');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('workflows.copySelection');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('gallery.galleryNavLeft');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('gallery.remix');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('viewer.deleteImage');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('app.invokeToOtherDestination');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('app.openProjectSwitcher');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('app.saveLayoutPreset');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('app.selectComposePreset');
  });

  it('uses the exported command-palette definition as the catalog entry', () => {
    expect(firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === 'app.openCommandPalette')).toBe(
      OPEN_COMMAND_PALETTE_HOTKEY
    );
  });

  it('keeps layout saving explicit and out of editable controls', () => {
    const saveLayout = firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === 'app.saveLayoutPreset');

    expect(saveLayout).toMatchObject({ allowInEditable: false, defaultKeys: [] });
  });
});
