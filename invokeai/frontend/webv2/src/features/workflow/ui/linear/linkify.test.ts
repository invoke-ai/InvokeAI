import { describe, expect, it } from 'vitest';

import { splitTextIntoLinkSegments } from './linkify';

describe('splitTextIntoLinkSegments', () => {
  it('keeps plain text as one segment', () => {
    expect(splitTextIntoLinkSegments('Pick a model.')).toEqual([{ kind: 'text', value: 'Pick a model.' }]);
    expect(splitTextIntoLinkSegments('')).toEqual([]);
  });

  it('separates URLs from the surrounding text without losing a character', () => {
    const text = 'Docs: https://invoke.ai/docs and http://example.com/a?b=1&c=2 — mirror.';

    expect(splitTextIntoLinkSegments(text)).toEqual([
      { kind: 'text', value: 'Docs: ' },
      { kind: 'link', value: 'https://invoke.ai/docs' },
      { kind: 'text', value: ' and ' },
      { kind: 'link', value: 'http://example.com/a?b=1&c=2' },
      { kind: 'text', value: ' — mirror.' },
    ]);
  });

  it('leaves sentence punctuation and unmatched brackets outside the link', () => {
    expect(splitTextIntoLinkSegments('(see https://a.b/c).')).toEqual([
      { kind: 'text', value: '(see ' },
      { kind: 'link', value: 'https://a.b/c' },
      { kind: 'text', value: ').' },
    ]);
    expect(splitTextIntoLinkSegments('https://en.wikipedia.org/wiki/Foo_(bar), ok')).toEqual([
      { kind: 'link', value: 'https://en.wikipedia.org/wiki/Foo_(bar)' },
      { kind: 'text', value: ', ok' },
    ]);
  });

  it('does not treat other schemes or bare domains as links', () => {
    expect(splitTextIntoLinkSegments('ftp://x.y or example.com or javascript:alert(1)')).toEqual([
      { kind: 'text', value: 'ftp://x.y or example.com or javascript:alert(1)' },
    ]);
  });
});
