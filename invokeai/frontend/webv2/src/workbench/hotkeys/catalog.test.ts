import { describe, expect, it } from 'vitest';

import {
  firstPartyHotkeyCatalog,
  getRegionFocusDefaultKey,
  OPEN_COMMAND_PALETTE_HOTKEY,
  regionFocusHotkeys,
} from './catalog';
import { IS_MAC_OS } from './keys';
import { toPlatformHotkey } from './resolve';

describe('firstPartyHotkeyCatalog', () => {
  it('keeps legacy default hotkey parity', () => {
    expect(firstPartyHotkeyCatalog).toHaveLength(121);
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

  it('clears the canvas on Mod+Alt+N, keeping the command id that saved bindings use', () => {
    const clearCanvas = firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === 'canvas.newSession');

    expect(clearCanvas).toMatchObject({ defaultKeys: ['mod+alt+n'], title: 'Clear Canvas' });
  });

  it('names the automate preset Workflow, keeping its id', () => {
    const selectWorkflow = firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === 'app.selectAutomatePreset');

    expect(selectWorkflow).toMatchObject({ defaultKeys: ['alt+4'], title: 'Select Workflow Preset' });
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

  it('offers crop all layers to frame unbound, for a key the user chooses', () => {
    const cropAll = firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === 'canvas.cropAllLayersToFrame');

    expect(cropAll).toMatchObject({ defaultKeys: [], implemented: true, title: 'Crop All Layers To Frame' });
  });

  it('binds transform to Shift+T only, leaving Mod+T to the browser', () => {
    const transform = firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === 'canvas.transformSelected');

    expect(transform).toMatchObject({ defaultKeys: ['shift+t'], implemented: true, title: 'Transform Selected Layer' });
  });

  it.each([
    ['canvas.setZoomTo100Percent', 'mod+1', 'Zoom to 100%'],
    ['canvas.setZoomTo200Percent', 'mod+2', 'Zoom to 200%'],
    ['canvas.setZoomTo400Percent', 'mod+3', 'Zoom to 400%'],
    ['canvas.setZoomTo800Percent', 'mod+4', 'Zoom to 800%'],
  ])('implements %s on %s, titled %s', (id, key, title) => {
    const zoom = firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === id);

    expect(zoom).toMatchObject({ defaultKeys: [key], implemented: true, title });
  });

  // Ids keep "bbox" so saved custom bindings survive; only the visible titles say "Frame".
  it('titles every frame command without the word bbox', () => {
    const titles = firstPartyHotkeyCatalog.map((hotkey) => hotkey.title);

    expect(titles.filter((title) => /bbox|bounding box/i.test(title))).toEqual([]);
    expect(firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === 'canvas.fitBboxToMasks')?.title).toBe(
      'Fit Frame To Masks'
    );
  });

  it('keeps layout saving explicit and out of editable controls', () => {
    const saveLayout = firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === 'app.saveLayoutPreset');

    expect(saveLayout).toMatchObject({ allowInEditable: false, defaultKeys: [] });
  });

  // On macOS, Option+Shift+Arrow selects text and Control+Option+Arrow is VoiceOver's navigation chord.
  it('moves region focus with Control+Command+Arrow on macOS and Alt+Shift+Arrow elsewhere', () => {
    expect(toPlatformHotkey(getRegionFocusDefaultKey('left', true), true)).toBe('ctrl+meta+arrowleft');
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
