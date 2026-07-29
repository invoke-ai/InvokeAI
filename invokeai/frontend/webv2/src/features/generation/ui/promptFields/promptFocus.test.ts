import { describe, expect, it } from 'vitest';

import type { PromptTriggerKey } from './promptFocus';

import { getActiveTriggerQuery, insertTextAtRange } from './promptFocus';

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

describe('getActiveTriggerQuery', () => {
  const query = (value: string, keys: PromptTriggerKey[] = ['<', '_']) =>
    getActiveTriggerQuery(value, value.length, keys);

  it('opens on the second underscore, with nothing typed yet', () => {
    expect(query('a photo of __')).toEqual({ key: '_', query: '', range: { end: 13, start: 11 } });
  });

  it('opens for a wildcard trigger adjacent to a word character', () => {
    expect(query('prefix__')).toEqual({ key: '_', query: '', range: { end: 8, start: 6 } });
  });

  it('narrows as the name is typed', () => {
    expect(query('a photo of __col')).toEqual({ key: '_', query: 'col', range: { end: 16, start: 11 } });
  });

  it('opens on an angle bracket and narrows the same way', () => {
    expect(query('a photo <emb')).toEqual({ key: '<', query: 'emb', range: { end: 12, start: 8 } });
  });

  it('takes the trigger nearest the caret', () => {
    expect(query('__one__ and <two')).toMatchObject({ key: '<', query: 'two' });
  });

  // The reference is finished; re-opening over it would fight the user.
  it('stays shut after a closed reference', () => {
    expect(query('a __colours__')).toBeNull();
    expect(query('a <embedding>')).toBeNull();
    expect(query('prefix__name__')).toBeNull();
  });

  it('stays shut after an empty closed wildcard reference', () => {
    expect(query('____')).toBeNull();
    expect(query('a ____')).toBeNull();
  });

  it('stays shut inside an ordinary word', () => {
    expect(query('a snake__case word')).toBeNull();
    expect(query('a photo of a cat')).toBeNull();
  });

  it('ends at a space, so a sentence does not keep it open', () => {
    expect(query('a __col our')).toBeNull();
  });

  // Embedding names routinely have spaces in them — `bad hands 5`, `Negative
  // Hand` — so ending the attempt at the first space put a whole class of them
  // out of reach inline, even though the filter would have matched.
  it('reads on through spaces inside an unclosed angle bracket', () => {
    expect(query('a photo of <bad hands')).toEqual({ key: '<', query: 'bad hands', range: { end: 21, start: 11 } });
  });

  it('still ends at a space when only wildcards are on offer', () => {
    expect(query('a photo of <bad hands', ['_'])).toBeNull();
  });

  it('stays shut once the angle bracket is closed', () => {
    expect(query('a <bad hands 5> photo of ')).toBeNull();
  });

  it('does not cross a newline looking for an angle bracket', () => {
    expect(query('a <bad\nhands')).toBeNull();
  });

  it('ends at a newline', () => {
    expect(query('a __col\nmore')).toBeNull();
  });

  it('keeps every valid wildcard-name character in range and closes over the limit', () => {
    const maximum = 'x'.repeat(128);

    expect(query(`a __${maximum}`)).toEqual({
      key: '_',
      query: maximum,
      range: { end: 132, start: 2 },
    });
    expect(query(`a __${maximum}x`)).toBeNull();
  });

  it('stays shut for malformed wildcard attempts', () => {
    expect(query('a __not a name')).toBeNull();
  });

  it('honours the keys the field answers to', () => {
    expect(query('a __col', ['<'])).toBeNull();
    expect(query('a <emb', ['_'])).toBeNull();
  });

  it('reads from the caret, not the end of the text', () => {
    expect(getActiveTriggerQuery('a __col and more', 7, ['_'])).toEqual({
      key: '_',
      query: 'col',
      range: { end: 7, start: 2 },
    });
  });
});
