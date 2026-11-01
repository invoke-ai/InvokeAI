import { describe, expect, it } from 'vitest';

import {
  firstPartyHotkeyCatalog,
  getRegionFocusDefaultKey,
  OPEN_COMMAND_PALETTE_HOTKEY,
  regionFocusHotkeys,
} from './catalog';
import { IS_MAC_OS } from './keys';
import { applyCustomHotkeys, resolveHotkey, toPlatformHotkey } from './resolve';

describe('firstPartyHotkeyCatalog', () => {
  it('keeps legacy default hotkey parity', () => {
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

    expect(openSettings).toMatchObject({ allowInEditable: true, defaultKeys: ['mod+,'] });
  });

  it('binds the color pair swap and reset to their classic keys', () => {
    const swap = firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === 'canvas.toggleFillColor');
    const reset = firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === 'canvas.setFillColorsToDefault');

    expect(swap).toMatchObject({ defaultKeys: ['x'] });
    expect(reset).toMatchObject({ defaultKeys: ['d'] });
  });

  it('offers crop all layers to frame unbound, for a key the user chooses', () => {
    const cropAll = firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === 'canvas.cropAllLayersToFrame');

    expect(cropAll).toMatchObject({ defaultKeys: [], title: 'Crop All Layers To Frame' });
  });

  it('binds transform to Shift+T only, leaving Mod+T to the browser', () => {
    const transform = firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === 'canvas.transformSelected');

    expect(transform).toMatchObject({ defaultKeys: ['shift+t'], title: 'Transform Selected Layer' });
  });

  it.each([
    ['canvas.setZoomTo100Percent', 'mod+1', 'Zoom to 100%'],
    ['canvas.setZoomTo200Percent', 'mod+2', 'Zoom to 200%'],
    ['canvas.setZoomTo400Percent', 'mod+3', 'Zoom to 400%'],
    ['canvas.setZoomTo800Percent', 'mod+4', 'Zoom to 800%'],
  ])('implements %s on %s, titled %s', (id, key, title) => {
    const zoom = firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === id);

    expect(zoom).toMatchObject({ defaultKeys: [key], title });
  });

  // Ids keep "bbox" so saved custom bindings survive; only the visible titles say "Frame".
  it('titles every frame command without the word bbox', () => {
    const titles = firstPartyHotkeyCatalog.map((hotkey) => hotkey.title);

    expect(titles.filter((title) => /bbox|bounding box/i.test(title))).toEqual([]);
    expect(firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === 'canvas.fitBboxToMasks')?.title).toBe(
      'Fit Frame To Masks'
    );
  });

  // Shift+R used to throw away the arrangement on a stray press; Mod+Shift+R is the browser's hard reload.
  it('leaves Shift+R alone and steps layout history on Mod+Alt+Z', () => {
    const registered = firstPartyHotkeyCatalog.map((hotkey) => applyCustomHotkeys(hotkey, {}));
    const context = {
      activeInstanceId: null,
      activeWidgetTypeId: null,
      focusedRegion: null,
      isModalPresent: false,
      projectId: 'project-1',
    } as const;
    const press = (matchedKey: string) =>
      resolveHotkey({ context, event: { target: null } as KeyboardEvent, hotkeys: registered, matchedKey })
        ?.commandId ?? null;

    expect(press('shift+r')).toBeNull();
    expect(press('mod+alt+z')).toBe('app.undoLayoutChange');
    expect(press('mod+alt+shift+z')).toBe('app.redoLayoutChange');
  });

  it('never lists one binding id twice', () => {
    const ids = firstPartyHotkeyCatalog.map((hotkey) => hotkey.id);

    expect(new Set(ids).size).toBe(ids.length);
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

    expect(toggle).toMatchObject({ defaultKeys: [] });
  });
});
