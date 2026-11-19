import { describe, expect, it } from 'vitest';

import { splitTextForMiddleTruncation } from './MiddleTruncate';

describe('splitTextForMiddleTruncation', () => {
  it('keeps the requested tail and puts the rest in the head', () => {
    expect(splitTextForMiddleTruncation('a-very-long-file-name.png', 8)).toEqual({
      head: 'a-very-long-file-',
      tail: 'name.png',
    });
  });

  it('puts short text entirely in the head so it degrades to end truncation', () => {
    expect(splitTextForMiddleTruncation('name.png', 8)).toEqual({ head: 'name.png', tail: '' });
    expect(splitTextForMiddleTruncation('x', 8)).toEqual({ head: 'x', tail: '' });
    expect(splitTextForMiddleTruncation('', 8)).toEqual({ head: '', tail: '' });
  });

  it('does not split emoji sequences at the boundary', () => {
    const family = '👨‍👩‍👧‍👦';
    const { head, tail } = splitTextForMiddleTruncation(`prefix-${family}${family}`, 1);

    expect(tail).toBe(family);
    expect(head).toBe(`prefix-${family}`);
  });

  it('treats a non-positive tail as plain end truncation', () => {
    expect(splitTextForMiddleTruncation('anything', 0)).toEqual({ head: 'anything', tail: '' });
  });
});
