import { ArrowRightToLineIcon, ChevronUpIcon, CommandIcon, CornerDownLeftIcon, OptionIcon } from 'lucide-react';
import { describe, expect, it } from 'vitest';

import { getShortcutKeyIcon } from './keyGlyphs';

describe('getShortcutKeyIcon', () => {
  it('gives universal keycap symbols an icon on every platform', () => {
    expect(getShortcutKeyIcon('enter', true)).toBe(CornerDownLeftIcon);
    expect(getShortcutKeyIcon('enter', false)).toBe(CornerDownLeftIcon);
    expect(getShortcutKeyIcon('tab', false)).toBe(ArrowRightToLineIcon);
  });

  it('gives modifier symbols to macOS only', () => {
    expect(getShortcutKeyIcon('cmd', true)).toBe(CommandIcon);
    expect(getShortcutKeyIcon('meta', true)).toBe(CommandIcon);
    expect(getShortcutKeyIcon('option', true)).toBe(OptionIcon);
    expect(getShortcutKeyIcon('ctrl', true)).toBe(ChevronUpIcon);
    expect(getShortcutKeyIcon('ctrl', false)).toBeNull();
    expect(getShortcutKeyIcon('shift', false)).toBeNull();
    expect(getShortcutKeyIcon('meta', false)).toBeNull();
  });

  it('leaves plain keys as text', () => {
    expect(getShortcutKeyIcon('k', true)).toBeNull();
    expect(getShortcutKeyIcon('esc', true)).toBeNull();
    expect(getShortcutKeyIcon('space', false)).toBeNull();
  });
});
