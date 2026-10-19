import { afterEach, describe, expect, it } from 'vitest';

import { isEditableHotkeyTarget } from './keys';

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
