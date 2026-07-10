import { describe, expect, it } from 'vitest';

import {
  areHotkeyStringsEquivalent,
  canonicalizeHotkeyString,
  formatHotkeyKeyForDisplay,
  formatHotkeyStringForPlatform,
  getHotkeyKeyFromEvent,
  getHotkeyStringAliases,
} from './hotkeyStrings';

describe('hotkeyStrings', () => {
  it('maps bracket key events to layout-independent physical-key tokens', () => {
    expect(getHotkeyKeyFromEvent('[', 'BracketLeft')).toBe('bracketleft');
    expect(getHotkeyKeyFromEvent(']', 'BracketRight')).toBe('bracketright');
    expect(getHotkeyKeyFromEvent('х', 'BracketLeft')).toBe('bracketleft');
    expect(getHotkeyKeyFromEvent('ъ', 'BracketRight')).toBe('bracketright');
  });

  it('does not infer physical punctuation keys without matching event codes', () => {
    expect(getHotkeyKeyFromEvent('[', 'Digit8')).toBe('[');
    expect(getHotkeyKeyFromEvent('[', undefined)).toBe('[');
    expect(canonicalizeHotkeyString('[')).toBe('[');
    expect(canonicalizeHotkeyString(']')).toBe(']');
    expect(canonicalizeHotkeyString('.')).toBe('.');
    expect(canonicalizeHotkeyString('Control+[', false)).toBe('mod+[');
    expect(canonicalizeHotkeyString('Meta+[', true)).toBe('mod+[');
  });

  it('formats physical bracket keys back to readable glyphs for display', () => {
    expect(formatHotkeyKeyForDisplay('bracketleft', false)).toBe('[');
    expect(formatHotkeyKeyForDisplay('bracketright', false)).toBe(']');
    expect(formatHotkeyKeyForDisplay('period', false)).toBe('.');
    expect(formatHotkeyStringForPlatform('mod+bracketleft', false)).toEqual(['ctrl', '[']);
    expect(formatHotkeyStringForPlatform('mod+bracketright', false)).toEqual(['ctrl', ']']);
    expect(formatHotkeyStringForPlatform('period', false)).toEqual(['.']);
  });

  it('uses the browser keyboard layout map when available', () => {
    const keyboardLayoutMap = new Map([
      ['BracketLeft', 'х'],
      ['BracketRight', 'ъ'],
    ]);

    expect(formatHotkeyKeyForDisplay('bracketleft', false, keyboardLayoutMap)).toBe('х');
    expect(formatHotkeyStringForPlatform('mod+bracketright', false, keyboardLayoutMap)).toEqual(['ctrl', 'ъ']);
  });

  it('expands physical and legacy glyph aliases for conflict detection', () => {
    expect(getHotkeyStringAliases('bracketleft', false)).toEqual(['bracketleft', '[']);
    expect(getHotkeyStringAliases('[', false)).toEqual(['[', 'bracketleft']);
    expect(getHotkeyStringAliases('shift+period', false)).toEqual(['shift+period', 'shift+.', 'shift+>']);
    expect(getHotkeyStringAliases('shift+>', false)).toEqual(['shift+>', 'shift+period', 'shift+.']);
    expect(areHotkeyStringsEquivalent('shift+>', 'shift+period', false)).toBe(true);
  });
});
