import { describe, expect, it } from 'vitest';

import { buildHotkeysData } from './useHotkeyData';

describe('buildHotkeysData', () => {
  const t = (key: string) => key;

  it('registers default merge hotkeys for canvas layers', () => {
    const hotkeysData = buildHotkeysData(t, {});
    const mergeDown = hotkeysData.canvas.hotkeys.mergeDown;
    const mergeVisible = hotkeysData.canvas.hotkeys.mergeVisible;

    expect(mergeDown).toBeDefined();
    expect(mergeVisible).toBeDefined();
    if (!mergeDown || !mergeVisible) {
      throw new Error('Expected merge hotkeys to be registered');
    }

    expect(mergeDown.defaultHotkeys).toEqual(['mod+e']);
    expect(mergeDown.hotkeys).toEqual(['mod+e']);
    expect(mergeVisible.defaultHotkeys).toEqual(['mod+shift+e']);
    expect(mergeVisible.hotkeys).toEqual(['mod+shift+e']);
  });

  it('applies custom hotkey overrides to merge actions', () => {
    const hotkeysData = buildHotkeysData(t, {
      'canvas.mergeDown': ['alt+m'],
      'canvas.mergeVisible': ['alt+shift+m'],
    });
    const mergeDown = hotkeysData.canvas.hotkeys.mergeDown;
    const mergeVisible = hotkeysData.canvas.hotkeys.mergeVisible;

    expect(mergeDown).toBeDefined();
    expect(mergeVisible).toBeDefined();
    if (!mergeDown || !mergeVisible) {
      throw new Error('Expected merge hotkeys to be registered');
    }

    expect(mergeDown.defaultHotkeys).toEqual(['mod+e']);
    expect(mergeDown.hotkeys).toEqual(['alt+m']);
    expect(mergeVisible.defaultHotkeys).toEqual(['mod+shift+e']);
    expect(mergeVisible.hotkeys).toEqual(['alt+shift+m']);
  });

  it('registers bracket tool-width hotkeys as layout-independent physical keys', () => {
    const hotkeysData = buildHotkeysData(t, {});
    const decrementToolWidth = hotkeysData.canvas.hotkeys.decrementToolWidth;
    const incrementToolWidth = hotkeysData.canvas.hotkeys.incrementToolWidth;
    const starImage = hotkeysData.gallery.hotkeys.starImage;

    expect(decrementToolWidth).toBeDefined();
    expect(incrementToolWidth).toBeDefined();
    expect(starImage).toBeDefined();
    if (!decrementToolWidth || !incrementToolWidth || !starImage) {
      throw new Error('Expected layout-sensitive punctuation hotkeys to be registered');
    }

    expect(decrementToolWidth.defaultHotkeys).toEqual(['bracketleft']);
    expect(decrementToolWidth.hotkeys).toEqual(['bracketleft']);
    expect(incrementToolWidth.defaultHotkeys).toEqual(['bracketright']);
    expect(incrementToolWidth.hotkeys).toEqual(['bracketright']);
    expect(starImage.defaultHotkeys).toEqual(['period']);
    expect(starImage.hotkeys).toEqual(['period']);
  });

  it('preserves custom punctuation glyph hotkeys without retargeting them to physical keys', () => {
    const hotkeysData = buildHotkeysData(t, {
      'canvas.decrementToolWidth': ['['],
      'canvas.incrementToolWidth': [']'],
    });

    expect(hotkeysData.canvas.hotkeys.decrementToolWidth?.hotkeys).toEqual(['[']);
    expect(hotkeysData.canvas.hotkeys.incrementToolWidth?.hotkeys).toEqual([']']);
  });

  it('formats physical hotkeys with the browser keyboard layout map when available', () => {
    const hotkeysData = buildHotkeysData(
      t,
      {},
      new Map([
        ['BracketLeft', 'х'],
        ['BracketRight', 'ъ'],
      ])
    );

    expect(hotkeysData.canvas.hotkeys.decrementToolWidth?.platformKeys).toEqual([['х']]);
    expect(hotkeysData.canvas.hotkeys.incrementToolWidth?.platformKeys).toEqual([['ъ']]);
  });
});
