import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { MiddleTruncate } from './MiddleTruncate';

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

describe('MiddleTruncate', () => {
  const LONG_NAME = 'a-very-long-image-file-name-with-a-meaningful-suffix-0042.png';

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

  /** The visible text, read from glyph positions: what is drawn inside the label's box, ellipsis aside. */

  it('re-measures a same-length text that draws at a different width', async () => {
    // Same length and tail; only the head's glyph widths differ.
    const narrow = 'iiiiiiii_0042.png';
    const wide = 'WWWWWWWW_0042.png';
    const subject = await renderAtWidth('80px', narrow);
    expect(subject.querySelectorAll('span')).toHaveLength(2);

    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <MiddleTruncate data-testid="subject" text={wide} />
        </ChakraProvider>
      );
    });

    expect(subject.querySelectorAll('span')).toHaveLength(1);
    expect(subject.textContent).toBe(wide);
  });
});
