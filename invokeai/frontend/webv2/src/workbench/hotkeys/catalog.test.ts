import { describe, expect, it } from 'vitest';

import {
  firstPartyHotkeyCatalog,
  getRegionFocusDefaultKey,
  OPEN_COMMAND_PALETTE_HOTKEY,
  regionFocusHotkeys,
} from './catalog';
import { IS_MAC_OS } from './keys';

describe('firstPartyHotkeyCatalog', () => {
  it('keeps legacy default hotkey parity', () => {
    expect(firstPartyHotkeyCatalog).toHaveLength(120);
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('app.togglePreview');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('app.invoke');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('app.openCommandPalette');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('canvas.mergeDown');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('canvas.newSession');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('workflows.copySelection');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('gallery.galleryNavLeft');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('gallery.remix');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('gallery.toggleStarredOnly');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('viewer.deleteImage');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('app.invokeToOtherDestination');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('app.openProjectSwitcher');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('app.saveLayoutPreset');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('app.selectComposePreset');
    expect(firstPartyHotkeyCatalog.map((hotkey) => hotkey.id)).toContain('app.selectVideoPreset');
  });

  it('uses the exported command-palette definition as the catalog entry', () => {
    expect(firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === 'app.openCommandPalette')).toBe(
      OPEN_COMMAND_PALETTE_HOTKEY
    );
  });

  it('binds settings to the platform-standard shortcut, reachable from a text field', () => {
    const openSettings = firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === 'app.openSettings');

    expect(openSettings).toMatchObject({ allowInEditable: true, defaultKeys: ['mod+,'], implemented: true });
  });

  it('marks the color pair swap and reset as implemented on their classic keys', () => {
    const swap = firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === 'canvas.toggleFillColor');
    const reset = firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === 'canvas.setFillColorsToDefault');

    expect(swap).toMatchObject({ defaultKeys: ['x'], implemented: true });
    expect(reset).toMatchObject({ defaultKeys: ['d'], implemented: true });
  });

  it('keeps layout saving explicit and out of editable controls', () => {
    const saveLayout = firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === 'app.saveLayoutPreset');

    expect(saveLayout).toMatchObject({ allowInEditable: false, defaultKeys: [] });
  });

  // Option+Shift+Arrow selects by word or paragraph in macOS text fields, where region focus must still be reachable.
  it('moves region focus with Control+Shift+Arrow on macOS and Alt+Shift+Arrow elsewhere', () => {
    expect(getRegionFocusDefaultKey('left', true)).toBe('ctrl+shift+arrowleft');
    expect(getRegionFocusDefaultKey('down', false)).toBe('alt+shift+arrowdown');
    expect(regionFocusHotkeys.map((hotkey) => hotkey.defaultKeys)).toEqual(
      (['left', 'right', 'up', 'down'] as const).map((direction) => [getRegionFocusDefaultKey(direction, IS_MAC_OS)])
    );
    expect(regionFocusHotkeys.every((hotkey) => hotkey.allowInEditable)).toBe(true);
  });

  // Cmd+Space is Spotlight on macOS; a modified Space on the focused thumbnail toggles it without a hotkey.
  it('ships the gallery focus toggle unbound but assignable', () => {
    const toggle = firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === 'gallery.toggleFocusedInSelection');

    expect(toggle).toMatchObject({ defaultKeys: [], implemented: true });
  });
});
