import { describe, expect, it } from 'vitest';

import { eventToHotkeyString, normalizeHotkeyString, toTinykeysBinding } from './keys';

describe('hotkey keys', () => {
  it('normalizes modifier order and aliases', () => {
    expect(normalizeHotkeyString('Shift+Mod+Enter')).toBe('mod+shift+enter');
    expect(normalizeHotkeyString('escape')).toBe('esc');
    expect(normalizeHotkeyString('Alt+ArrowUp')).toBe('alt+arrowup');
  });

  it('converts normalized keys to tinykeys syntax', () => {
    expect(toTinykeysBinding('mod+enter')).toBe('$mod+Enter');
    expect(toTinykeysBinding('alt+]')).toBe('Alt+BracketRight');
    expect(toTinykeysBinding('.')).toBe('Period');
  });

  it('ignores IME composition events when recording hotkeys', () => {
    expect(eventToHotkeyString({ isComposing: true, key: 'a' } as KeyboardEvent)).toBe('');
    expect(eventToHotkeyString({ key: 'Process', keyCode: 229 } as KeyboardEvent)).toBe('');
  });
});
