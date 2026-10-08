import { describe, expect, it } from 'vitest';

import { getRegionFocusDefaultKey } from './catalog';
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

  it('records both Control and Command, so the macOS region-focus default can be recorded back', () => {
    const event = { ctrlKey: true, key: 'ArrowLeft', metaKey: true } as KeyboardEvent;

    expect(eventToHotkeyString(event, true)).toBe(normalizeHotkeyString(getRegionFocusDefaultKey('left', true)));
    expect(eventToHotkeyString({ ctrlKey: true, key: 'ArrowLeft' } as KeyboardEvent, true)).toBe('ctrl+arrowleft');
  });

  it('records both Control and Meta outside macOS, with Control as the primary modifier', () => {
    const event = { ctrlKey: true, key: 'ArrowLeft', metaKey: true } as KeyboardEvent;

    expect(eventToHotkeyString(event, false)).toBe('mod+meta+arrowleft');
    expect(eventToHotkeyString({ key: 'ArrowLeft', metaKey: true } as KeyboardEvent, false)).toBe('meta+arrowleft');
  });
});
