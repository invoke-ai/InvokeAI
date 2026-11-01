import { afterEach, describe, expect, it } from 'vitest';

import { editTextareaText, isOtherTextEntry } from './promptFocus';

const mount = (html: string) => {
  const host = document.createElement('div');
  host.innerHTML = html;
  document.body.append(host);
  return host;
};

afterEach(() => {
  document.body.replaceChildren();
});

describe('editTextareaText', () => {
  it('edits through the browser, so its own undo restores the text', () => {
    const host = mount('<textarea>a cat</textarea>');
    const textarea = host.querySelector('textarea')!;

    expect(editTextareaText(textarea, 'a dog')).toBe(true);
    expect(textarea.value).toBe('a dog');

    document.execCommand('undo');
    expect(textarea.value).toBe('a cat');
  });

  // A hidden view (an inactive dock tab, a shaded window) cannot take focus; typing there would land elsewhere.
  it('refuses a textarea that cannot take focus, leaving the focused field untouched', () => {
    const host = mount(
      '<div data-scope="popover" data-part="content"><input value="search"></div>' +
        '<div style="display: none"><textarea>a cat</textarea></div>'
    );
    const input = host.querySelector('input')!;
    const textarea = host.querySelector('textarea')!;

    input.focus();

    expect(editTextareaText(textarea, 'EXPANDED')).toBe(false);
    expect(input.value).toBe('search');
    expect(textarea.value).toBe('a cat');
  });

  it('refuses while the user types in another field, so it never takes focus from them', () => {
    const host = mount('<textarea id="prompt">a cat</textarea><input id="other">');
    const textarea = host.querySelector<HTMLTextAreaElement>('#prompt')!;
    const other = host.querySelector<HTMLInputElement>('#other')!;

    other.focus();

    expect(editTextareaText(textarea, 'a dog')).toBe(false);
    expect(document.activeElement).toBe(other);
    expect(textarea.value).toBe('a cat');
  });
});

describe('isOtherTextEntry', () => {
  it('counts fields the user types in outside popovers, not the prompt or controls', () => {
    const host = mount(
      '<textarea id="prompt"></textarea><textarea id="negative"></textarea><input id="name">' +
        '<input id="check" type="checkbox"><button id="button">Go</button><div id="editable" contenteditable="true"></div>' +
        '<div data-scope="popover" data-part="content"><input id="search"></div>'
    );
    const prompt = host.querySelector<HTMLTextAreaElement>('#prompt')!;
    const byId = (id: string) => host.querySelector(`#${id}`);

    expect(isOtherTextEntry(byId('negative'), prompt)).toBe(true);
    expect(isOtherTextEntry(byId('name'), prompt)).toBe(true);
    expect(isOtherTextEntry(byId('editable'), prompt)).toBe(true);
    expect(isOtherTextEntry(prompt, prompt)).toBe(false);
    expect(isOtherTextEntry(byId('check'), prompt)).toBe(false);
    expect(isOtherTextEntry(byId('button'), prompt)).toBe(false);
    expect(isOtherTextEntry(byId('search'), prompt)).toBe(false);
  });
});
