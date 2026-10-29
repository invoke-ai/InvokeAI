import { describe, expect, it } from 'vitest';

import { getRegionFocusDefaultKey } from './catalog';
import {
  eventToHotkeyString,
  formatHotkeyAriaLabel,
  formatHotkeyForPlatform,
  acceptsHotkeyPress,
  formatHotkeyLabel,
  isHotkeyPress,
  normalizeHotkeyString,
  toTinykeysBinding,
} from './keys';

/** A keydown as a browser reports it, with the modifier state tinykeys reads. */
const press = (init: { key: string; code: string; altKey?: boolean; metaKey?: boolean }): KeyboardEvent =>
  ({
    altKey: false,
    ctrlKey: false,
    metaKey: false,
    shiftKey: false,
    ...init,
    getModifierState: (modifier: string) =>
      (modifier === 'Alt' && init.altKey === true) || (modifier === 'Meta' && init.metaKey === true),
  }) as unknown as KeyboardEvent;

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

  it('names an Alt chord with Cmd, Ctrl or Meta by its character or its position, every other binding by character', () => {
    expect(toTinykeysBinding('mod+alt+n')).toBe('$mod+Alt+(n|KeyN)');
    expect(toTinykeysBinding('ctrl+alt+1')).toBe('Control+Alt+(1|Digit1)');
    expect(toTinykeysBinding('alt+a')).toBe('Alt+a');
    expect(toTinykeysBinding('alt+1')).toBe('Alt+1');
    expect(toTinykeysBinding('mod+n')).toBe('$mod+n');
    expect(toTinykeysBinding('alt+]')).toBe('Alt+BracketRight');
  });

  it('fires Cmd+Option+N on macOS, where Option turns the key into a dead key', () => {
    expect(isHotkeyPress(press({ altKey: true, code: 'KeyN', key: 'Dead', metaKey: true }), 'meta+alt+n')).toBe(true);
    expect(isHotkeyPress(press({ altKey: true, code: 'KeyN', key: '˜', metaKey: true }), 'meta+alt+n')).toBe(true);
    expect(isHotkeyPress(press({ altKey: true, code: 'KeyM', key: 'µ', metaKey: true }), 'meta+alt+n')).toBe(false);
  });

  it('leaves Option+A to type å: an Alt chord alone matches only its character', () => {
    expect(isHotkeyPress(press({ altKey: true, code: 'KeyA', key: 'å' }), 'alt+a')).toBe(false);
    expect(isHotkeyPress(press({ altKey: true, code: 'KeyA', key: 'a' }), 'alt+a')).toBe(true);
  });

  it('keeps a letter on the key that types it on other layouts', () => {
    // AZERTY: the key labelled A sits where QWERTY has Q.
    expect(isHotkeyPress(press({ altKey: true, code: 'KeyQ', key: 'a' }), 'alt+a')).toBe(true);
    expect(isHotkeyPress(press({ altKey: true, code: 'KeyA', key: 'q' }), 'alt+a')).toBe(false);
    // Dvorak: the key at QWERTY N types b; with Cmd+Alt it is still b, not n.
    expect(isHotkeyPress(press({ altKey: true, code: 'KeyN', key: 'b', metaKey: true }), 'meta+alt+n')).toBe(false);
    // On a Cyrillic layout, the key at N's position types т: the plain `n` binding is not it.
    expect(isHotkeyPress(press({ code: 'KeyN', key: 'т' }), 'n')).toBe(false);
  });

  it('reads AltGr as Alt, as tinykeys does, except for a press matched by position', () => {
    // Browsers report AltGr for Ctrl+Alt on AltGr layouts.
    const withAltGr = (init: Parameters<typeof press>[0] & { ctrlKey?: boolean }) =>
      ({
        ...press(init),
        ctrlKey: init.ctrlKey === true,
        getModifierState: (modifier: string) =>
          modifier === 'AltGraph' ||
          modifier === 'Alt' ||
          (modifier === 'Control' && init.ctrlKey === true) ||
          (modifier === 'Meta' && init.metaKey === true),
      }) as unknown as KeyboardEvent;

    expect(
      acceptsHotkeyPress(withAltGr({ altKey: true, code: 'Enter', ctrlKey: true, key: 'Enter' }), 'alt+mod+enter')
    ).toBe(true);
    expect(acceptsHotkeyPress(withAltGr({ altKey: true, code: 'Digit1', key: '1' }), 'alt+1')).toBe(true);
    expect(isHotkeyPress(withAltGr({ altKey: true, code: 'Digit1', key: '1' }), 'alt+1')).toBe(true);
    // AltGr+N types ń: not Cmd/Ctrl+Alt+N by position.
    expect(isHotkeyPress(withAltGr({ altKey: true, code: 'KeyN', key: 'ń', metaKey: true }), 'meta+alt+n')).toBe(false);
  });

  it('records a typed letter as typed, and a dead key or symbol by its position', () => {
    expect(eventToHotkeyString(press({ altKey: true, code: 'KeyN', key: 'Dead', metaKey: true }), true)).toBe(
      'mod+alt+n'
    );
    expect(eventToHotkeyString(press({ altKey: true, code: 'KeyQ', key: 'a' }), false)).toBe('alt+a');
    // Ctrl+Alt+N on a Polish layout is AltGr+N, which types ń: recorded as typed, it matches again.
    const polish = {
      ...press({ altKey: true, code: 'KeyN', key: 'ń' }),
      ctrlKey: true,
      getModifierState: (modifier: string) => modifier === 'AltGraph' || modifier === 'Alt' || modifier === 'Control',
    } as unknown as KeyboardEvent;
    expect(eventToHotkeyString(polish, false)).toBe('mod+alt+ń');
    expect(acceptsHotkeyPress(polish, 'mod+alt+ń')).toBe(true);
    // Option alone matches only by character, so it records the character it types.
    expect(eventToHotkeyString(press({ altKey: true, code: 'KeyA', key: 'å' }), true)).toBe('alt+å');
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

  it('shows macOS modifiers in Control, Option, Shift, Command order and leaves other platforms alone', () => {
    expect(formatHotkeyForPlatform(getRegionFocusDefaultKey('left', true), true)).toEqual(['ctrl', 'cmd', 'arrowleft']);
    expect(formatHotkeyForPlatform('mod+shift+alt+ctrl+z', true)).toEqual(['ctrl', 'option', 'shift', 'cmd', 'z']);
    expect(formatHotkeyForPlatform('mod+shift+alt+z', false)).toEqual(['ctrl', 'shift', 'alt', 'z']);
  });

  it('labels a binding as adjacent glyphs on macOS and +-joined words elsewhere', () => {
    expect(formatHotkeyLabel('mod+enter', true)).toBe('⌘↵');
    expect(formatHotkeyLabel('mod+k', true)).toBe('⌘K');
    expect(formatHotkeyLabel('alt+mod+enter', true)).toBe('⌥⌘↵');
    expect(formatHotkeyLabel('mod+enter', false)).toBe('Ctrl+Enter');
    expect(formatHotkeyLabel('alt+mod+enter', false)).toBe('Ctrl+Alt+Enter');
  });

  it('formats a binding for aria-keyshortcuts', () => {
    expect(formatHotkeyAriaLabel('mod+enter', true)).toBe('Meta+Enter');
    expect(formatHotkeyAriaLabel('alt+mod+s', true)).toBe('Alt+Meta+S');
    expect(formatHotkeyAriaLabel('alt+mod+s', false)).toBe('Control+Alt+S');
  });
});
