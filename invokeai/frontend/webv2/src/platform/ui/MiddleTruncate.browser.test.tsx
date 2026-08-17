import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { MiddleTruncate, splitTextForMiddleTruncation } from './MiddleTruncate';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const renderAtWidth = async (width: string, text: string): Promise<HTMLElement> => {
  host = document.createElement('div');
  host.style.cssText = `width:${width};`;
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <MiddleTruncate data-testid="subject" text={text} />
      </ChakraProvider>
    );
  });

  return host.querySelector<HTMLElement>('[data-testid="subject"]')!;
};

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

describe('MiddleTruncate', () => {
  const LONG_NAME = 'a-very-long-image-file-name-with-a-meaningful-suffix-0042.png';

  it('renders the full text seamlessly when it fits', async () => {
    const subject = await renderAtWidth('40rem', LONG_NAME);

    expect(subject.textContent).toBe(LONG_NAME);
    expect(subject.scrollWidth).toBeLessThanOrEqual(subject.clientWidth);
  });

  it('keeps the tail fully visible and ellipsizes the head when constrained', async () => {
    const subject = await renderAtWidth('12rem', LONG_NAME);
    const [head, tail] = subject.querySelectorAll<HTMLElement>('span');

    // The full string stays in the DOM for selection, copy, and a11y.
    expect(subject.textContent).toBe(LONG_NAME);
    expect(tail.textContent).toBe('0042.png');
    // The head overflows (that is where the ellipsis lives)...
    expect(head.scrollWidth).toBeGreaterThan(head.clientWidth);
    // ...while the tail is not clipped at all.
    expect(tail.scrollWidth).toBeLessThanOrEqual(tail.clientWidth);
    expect(tail.getBoundingClientRect().right).toBeLessThanOrEqual(subject.getBoundingClientRect().right + 0.5);
  });

  it('exposes the full text as a native tooltip', async () => {
    const subject = await renderAtWidth('12rem', LONG_NAME);

    expect(subject.getAttribute('title')).toBe(LONG_NAME);
  });

  it('preserves a space that lands at the split point', async () => {
    const subject = await renderAtWidth('8rem', 'Panther and the Flask');
    const [head, tail] = subject.querySelectorAll<HTMLElement>('span');

    expect(head.textContent).toBe('Panther and t');
    expect(tail.textContent).toBe('he Flask');
    // `white-space: pre` keeps the leading space from collapsing; with
    // `nowrap` the flex item would strip it and fuse the halves.
    expect(getComputedStyle(tail).whiteSpace).toBe('pre');
  });
});
