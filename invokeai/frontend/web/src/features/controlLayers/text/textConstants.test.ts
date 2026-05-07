import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  getFontStackById,
  getTextFontStack,
  setCustomTextFontStacks,
  subscribeToCustomTextFontStacks,
  TEXT_FONT_STACKS,
} from './textConstants';

describe('textConstants custom font registry', () => {
  afterEach(() => {
    setCustomTextFontStacks([]);
  });

  it('notifies subscribers when custom fonts are updated', () => {
    const listener = vi.fn();
    const unsubscribe = subscribeToCustomTextFontStacks(listener);

    setCustomTextFontStacks([{ id: 'user:fonts/MyFont-Regular.ttf', label: 'My Font', stack: '"My Font",sans-serif' }]);

    expect(listener).toHaveBeenCalledTimes(1);
    expect(getFontStackById('user:fonts/MyFont-Regular.ttf')).toBe('"My Font",sans-serif');

    unsubscribe();
  });

  it('distinguishes missing custom font ids from known font stacks', () => {
    expect(getTextFontStack('user:missing-font')).toBeUndefined();
    expect(getFontStackById('user:missing-font')).toBe(TEXT_FONT_STACKS[0]?.stack ?? 'sans-serif');
  });
});
