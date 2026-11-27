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
  const maximum = 'x'.repeat(128);
  type Row = { caret?: number; keys?: PromptTriggerKey[]; name: string; text: string };
  const read = ({ caret, keys = ['<', '_'], text }: Row) => getActiveTriggerQuery(text, caret ?? text.length, keys);

  it.each<Row & { expected: ReturnType<typeof getActiveTriggerQuery> }>([
    {
      name: 'opens on the second underscore, with nothing typed yet',
      text: 'a photo of __',
      expected: { key: '_', query: '', range: { end: 13, start: 11 } },
    },
    {
      name: 'opens for a wildcard trigger adjacent to a word character',
      text: 'prefix__',
      expected: { key: '_', query: '', range: { end: 8, start: 6 } },
    },
    {
      name: 'narrows as the name is typed',
      text: 'a photo of __col',
      expected: { key: '_', query: 'col', range: { end: 16, start: 11 } },
    },
    {
      name: 'opens on an angle bracket and narrows the same way',
      text: 'a photo <emb',
      expected: { key: '<', query: 'emb', range: { end: 12, start: 8 } },
    },
    {
      name: 'takes the trigger nearest the caret',
      text: '__one__ and <two',
      expected: { key: '<', query: 'two', range: { end: 16, start: 12 } },
    },
    {
      name: 'opens a new wildcard trigger immediately after a completed reference',
      text: '__a____',
      expected: { key: '_', query: '', range: { end: 7, start: 5 } },
    },
    {
      name: 'reads on through spaces inside an unclosed angle bracket',
      text: 'a photo of <bad hands',
      expected: { key: '<', query: 'bad hands', range: { end: 21, start: 11 } },
    },
    {
      name: 'keeps every valid wildcard-name character in range',
      text: `a __${maximum}`,
      expected: { key: '_', query: maximum, range: { end: 132, start: 2 } },
    },
    {
      name: 'reads from the caret, not the end of the text',
      caret: 7,
      keys: ['_'],
      text: 'a __col and more',
      expected: { key: '_', query: 'col', range: { end: 7, start: 2 } },
    },
  ])('$name', ({ expected, ...row }) => {
    expect(read(row)).toEqual(expected);
  });

  it.each<Row>([
    { name: 'stays shut after a closed wildcard reference', text: 'a __colours__' },
    { name: 'stays shut after a closed embedding reference', text: 'a <embedding>' },
    { name: 'stays shut after a closed reference adjacent to a word', text: 'prefix__name__' },
    { name: 'stays shut after an empty closed wildcard reference', text: '____' },
    { name: 'stays shut after an empty closed wildcard reference with text before it', text: 'a ____' },
    { name: 'stays shut inside an ordinary word', text: 'a snake__case word' },
    { name: 'stays shut in plain text', text: 'a photo of a cat' },
    { name: 'ends at a space, so a sentence does not keep it open', text: 'a __col our' },
    { name: 'still ends at a space when only wildcards are on offer', keys: ['_'], text: 'a photo of <bad hands' },
    { name: 'stays shut once the angle bracket is closed', text: 'a <bad hands 5> photo of ' },
    { name: 'does not cross a newline looking for an angle bracket', text: 'a <bad\nhands' },
    { name: 'ends at a newline', text: 'a __col\nmore' },
    { name: 'closes over the wildcard-name length limit', text: `a __${maximum}x` },
    { name: 'stays shut for malformed wildcard attempts', text: 'a __not a name' },
    { name: 'stays shut for a wildcard when the field answers only to <', keys: ['<'], text: 'a __col' },
    { name: 'stays shut for an embedding when the field answers only to _', keys: ['_'], text: 'a <emb' },
  ])('$name', (row) => {
    expect(read(row)).toBeNull();
  });
});
