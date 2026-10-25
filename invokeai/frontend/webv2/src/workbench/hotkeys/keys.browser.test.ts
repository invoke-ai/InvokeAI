import { afterEach, describe, expect, it } from 'vitest';

import { isEditableHotkeyTarget, isNativeControlKeyPress } from './keys';

describe('isEditableHotkeyTarget', () => {
  const host = document.createElement('div');

  document.body.append(host);

  afterEach(() => {
    host.replaceChildren();
  });

  const mount = (html: string): HTMLElement => {
    host.innerHTML = html;

    return host.querySelector<HTMLElement>('[data-target]')!;
  };

  it('protects the native shortcuts of anything that takes typed text', () => {
    expect(isEditableHotkeyTarget(mount('<input data-target />'))).toBe(true);
    expect(isEditableHotkeyTarget(mount('<input data-target type="number" />'))).toBe(true);
    expect(isEditableHotkeyTarget(mount('<textarea data-target></textarea>'))).toBe(true);
    expect(isEditableHotkeyTarget(mount('<select data-target></select>'))).toBe(true);
    expect(isEditableHotkeyTarget(mount('<div contenteditable="true"><span data-target>x</span></div>'))).toBe(true);
  });

  it('lets shortcuts through from a focused switch, checkbox, or button', () => {
    // A workflow node's boolean switch is a hidden checkbox: Ctrl+Z from it must reach the editor's undo.
    expect(isEditableHotkeyTarget(mount('<label><input data-target type="checkbox" /></label>'))).toBe(false);
    expect(isEditableHotkeyTarget(mount('<input data-target type="radio" />'))).toBe(false);
    expect(isEditableHotkeyTarget(mount('<input data-target type="range" />'))).toBe(false);
    expect(isEditableHotkeyTarget(mount('<button data-target type="button">x</button>'))).toBe(false);
    expect(isEditableHotkeyTarget(null)).toBe(false);
  });
});

describe('isNativeControlKeyPress', () => {
  const host = document.createElement('div');

  document.body.append(host);

  afterEach(() => {
    host.replaceChildren();
  });

  const press = (html: string, init: KeyboardEventInit): boolean => {
    host.innerHTML = html;
    const event = new KeyboardEvent('keydown', init);
    host.querySelector('[data-target]')!.dispatchEvent(event);

    return isNativeControlKeyPress(event);
  };

  it('leaves arrows, Home and End to a focused radio group or slider', () => {
    expect(press('<input data-target type="radio" />', { key: 'ArrowRight' })).toBe(true);
    expect(press('<input data-target type="radio" />', { key: 'Home' })).toBe(true);
    expect(press('<input data-target type="range" />', { key: 'ArrowUp' })).toBe(true);
    expect(press('<div role="radiogroup"><span data-target>x</span></div>', { key: 'End' })).toBe(true);
    expect(press('<div data-target role="slider" tabindex="0"></div>', { key: 'ArrowLeft' })).toBe(true);
  });

  it('leaves other keys and chords to shortcuts', () => {
    expect(press('<input data-target type="radio" />', { key: 'a' })).toBe(false);
    expect(press('<input data-target type="radio" />', { ctrlKey: true, key: 'ArrowRight' })).toBe(false);
    expect(press('<input data-target type="radio" />', { altKey: true, key: 'ArrowRight' })).toBe(false);
  });

  it('is not about text fields, which isEditableHotkeyTarget covers', () => {
    expect(press('<input data-target />', { key: 'ArrowRight' })).toBe(false);
    expect(press('<button data-target type="button">x</button>', { key: 'ArrowRight' })).toBe(false);
  });
});
