import { describe, expect, it } from 'vitest';

import { isHotkeyModalLayerActive, registerHotkeyModalLayer } from './modalLayer';

describe('hotkey modal layer registry', () => {
  it('is active only while a registered layer is mounted', () => {
    expect(isHotkeyModalLayerActive()).toBe(false);

    const unregister = registerHotkeyModalLayer('settings');

    expect(isHotkeyModalLayerActive()).toBe(true);

    unregister();

    expect(isHotkeyModalLayerActive()).toBe(false);
  });
});
