import { describe, expect, it } from 'vitest';

import type { PromptTriggerKey } from './promptFocus';

import { getPromptTriggerRange, insertTextAtRange } from './promptFocus';

describe('insertTextAtRange', () => {
  it('inserts text at a collapsed range', () => {
    expect(insertTextAtRange('a  cat', 'fluffy', { end: 2, start: 2 })).toEqual({ caret: 8, value: 'a fluffy cat' });
  });

  it('replaces the selected range with text', () => {
    expect(insertTextAtRange('a < cat', 'fluffy', { end: 3, start: 2 })).toEqual({ caret: 8, value: 'a fluffy cat' });
  });

  it('clamps stale ranges to the prompt bounds', () => {
    expect(insertTextAtRange('cat', ' fluffy', { end: 99, start: 99 })).toEqual({ caret: 10, value: 'cat fluffy' });
  });

  it('uses the textarea caret when no explicit range is provided', () => {
    expect(insertTextAtRange('cat', ' fluffy', undefined, 3)).toEqual({ caret: 10, value: 'cat fluffy' });
  });
});

describe('getPromptTriggerRange', () => {
  // The caret sits at the end of `value` unless a test says otherwise.
  const match = (value: string, key: string, keys: readonly PromptTriggerKey[] = ['<', '_']) =>
    getPromptTriggerRange(value, value.length, value.length, key, keys);

  it('opens on `<`, replacing nothing', () => {
    expect(match('a photo of ', '<')).toEqual({ key: '<', range: { end: 11, start: 11 } });
  });

  it('opens on the second underscore of a reference, replacing the first', () => {
    expect(match('a photo of _', '_')).toEqual({ key: '_', range: { end: 12, start: 11 } });
    expect(match('_', '_')).toEqual({ key: '_', range: { end: 1, start: 0 } });
    // Punctuation is a boundary too — `(__style__)1.2` is ordinary usage.
    expect(match('(_', '_')).toEqual({ key: '_', range: { end: 2, start: 1 } });
  });

  // Regression: this swallowed the keystroke, so `__` could not be typed at all.
  it('stays shut in the middle of a word', () => {
    expect(match('snake_', '_')).toBeNull();
    expect(match('close_up_', '_')).toBeNull();
    expect(match('2_', '_')).toBeNull();
  });

  it('stays shut on the first underscore', () => {
    expect(match('a photo of', '_')).toBeNull();
  });

  it('ignores keys the field does not answer to', () => {
    expect(match('a photo of _', '_', ['<'])).toBeNull();
    expect(match('a photo of ', '<', ['_'])).toBeNull();
  });

  it('ignores every other key', () => {
    expect(match('a photo of ', 'a')).toBeNull();
    expect(match('a photo of ', 'Enter')).toBeNull();
  });

  it('replaces the whole selection, not just the caret', () => {
    // `a _[cat]` with `cat` selected: typing `_` would overwrite the selection,
    // so the picked reference has to take the underscore and the selection with it.
    expect(getPromptTriggerRange('a _cat', 3, 6, '_', ['_'])).toEqual({ key: '_', range: { end: 6, start: 2 } });
  });
});
