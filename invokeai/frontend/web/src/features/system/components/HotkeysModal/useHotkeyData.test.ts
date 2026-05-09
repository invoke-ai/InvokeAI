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
});
