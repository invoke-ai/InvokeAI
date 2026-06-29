import { describe, expect, it } from 'vitest';

import { insertTextAtRange } from './promptFocus';

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
