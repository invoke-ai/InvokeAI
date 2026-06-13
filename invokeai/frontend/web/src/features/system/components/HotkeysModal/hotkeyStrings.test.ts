import { describe, expect, it } from 'vitest';

import {
  canonicalizeHotkeyString,
  formatHotkeyKeyForDisplay,
  formatHotkeyStringForPlatform,
  getHotkeyKeyFromEvent,
} from './hotkeyStrings';

describe('hotkeyStrings', () => {
  it('maps bracket key events to layout-independent physical-key tokens', () => {
    expect(getHotkeyKeyFromEvent('[', 'BracketLeft')).toBe('bracketleft');
    expect(getHotkeyKeyFromEvent(']', 'BracketRight')).toBe('bracketright');
    expect(getHotkeyKeyFromEvent('х', 'BracketLeft')).toBe('bracketleft');
    expect(getHotkeyKeyFromEvent('ъ', 'BracketRight')).toBe('bracketright');
  });

  it('canonicalizes legacy literal bracket hotkeys', () => {
    expect(canonicalizeHotkeyString('[')).toBe('bracketleft');
    expect(canonicalizeHotkeyString(']')).toBe('bracketright');
    expect(canonicalizeHotkeyString('.')).toBe('period');
    expect(canonicalizeHotkeyString('mod+[')).toBe('mod+bracketleft');
    expect(canonicalizeHotkeyString('mod+]')).toBe('mod+bracketright');
  });

  it('formats physical bracket keys back to readable glyphs for display', () => {
    expect(formatHotkeyKeyForDisplay('bracketleft', false)).toBe('[');
    expect(formatHotkeyKeyForDisplay('bracketright', false)).toBe(']');
    expect(formatHotkeyKeyForDisplay('period', false)).toBe('.');
    expect(formatHotkeyStringForPlatform('mod+bracketleft', false)).toEqual(['ctrl', '[']);
    expect(formatHotkeyStringForPlatform('mod+bracketright', false)).toEqual(['ctrl', ']']);
    expect(formatHotkeyStringForPlatform('period', false)).toEqual(['.']);
  });
});
