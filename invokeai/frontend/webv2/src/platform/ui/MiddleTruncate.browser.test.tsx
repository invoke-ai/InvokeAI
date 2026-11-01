import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';
import { page } from 'vitest/browser';

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

  /** The visible text, read from glyph positions: what is drawn inside the label's box, ellipsis aside. */
  const visibleGraphemes = (subject: HTMLElement): string => {
    const bounds = subject.getBoundingClientRect();
    let visible = '';

    for (const span of subject.querySelectorAll('span')) {
      const node = span.firstChild;
      const spanBounds = span.getBoundingClientRect();

      if (!(node instanceof Text)) {
        continue;
      }

      for (let index = 0; index < node.length; index += 1) {
        const range = document.createRange();
        range.setStart(node, index);
        range.setEnd(node, index + 1);
        const glyph = range.getBoundingClientRect();
        // A flex head clips its own overflow; the root clips everything.
        const right =
          getComputedStyle(span).overflow === 'hidden' ? Math.min(bounds.right, spanBounds.right) : bounds.right;

        if (glyph.right <= right + 0.5) {
          visible += node.data[index];
        }
      }
    }

    return visible;
  };

  it.each([
    ['about 33px', '33px'],
    ['about 60px', '60px'],
  ])('ends the whole text in an ellipsis at %s, where the tail would be a clipped fragment', async (_label, width) => {
    const subject = await renderAtWidth(width, 'Uncategorized');
    const spans = subject.querySelectorAll<HTMLElement>('span');

    // One ellipsized span holds the whole text, so nothing is clipped without an ellipsis.
    expect(spans).toHaveLength(1);
    expect(getComputedStyle(spans[0]!).textOverflow).toBe('ellipsis');
    expect(spans[0]!.scrollWidth).toBeGreaterThan(spans[0]!.clientWidth);
    // The start shows, not a slice from the middle of the tail.
    expect(visibleGraphemes(subject).startsWith('U')).toBe(true);
    expect('Uncategorized'.startsWith(visibleGraphemes(subject))).toBe(true);
    // The full text stays available as before.
    expect(subject.textContent).toBe('Uncategorized');
    expect(subject.getAttribute('title')).toBe('Uncategorized');
  });

  it('returns to the middle form once the width holds the tail and a few head graphemes', async () => {
    const subject = await renderAtWidth('33px', LONG_NAME);
    expect(subject.querySelectorAll('span')).toHaveLength(1);

    await act(() => {
      host!.style.width = '12rem';
    });
    await expect.poll(() => subject.querySelectorAll('span').length).toBe(2);

    const [head] = subject.querySelectorAll<HTMLElement>('span');
    expect(head.scrollWidth).toBeGreaterThan(head.clientWidth);
    expect(visibleGraphemes(subject).endsWith('0042.png')).toBe(true);
  });

  it('keeps the middle form unchanged at a comfortable width', async () => {
    const subject = await renderAtWidth('12rem', LONG_NAME);
    const visible = visibleGraphemes(subject);

    expect(subject.querySelectorAll('span')).toHaveLength(2);
    expect(visible.startsWith('a-very')).toBe(true);
    expect(visible.endsWith('0042.png')).toBe(true);
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
    // white-space: pre preserves the leading space at the flex split.
    expect(getComputedStyle(tail).whiteSpace).toBe('pre');
  });

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

  it.each([
    ['split', '12rem'],
    ['end-truncated', '33px'],
  ])('gives a named control the whole text once in the %s form, keeping its text for copy', async (_form, width) => {
    host = document.createElement('div');
    host.style.cssText = `width:${width};`;
    document.body.append(host);
    root = createRoot(host);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <button style={{ display: 'block', width: '100%' }} type="button">
            <MiddleTruncate text={LONG_NAME} />
          </button>
        </ChakraProvider>
      );
    });

    // Without this, the split's flex items read as two words.
    await expect.element(page.getByRole('button', { exact: true, name: LONG_NAME })).toBeInTheDocument();
    expect(host.querySelector('button')!.textContent).toBe(LONG_NAME);
    const label = host.querySelector<HTMLElement>('[title]')!;
    expect(label.getAttribute('title')).toBe(LONG_NAME);
    expect(label.dataset.fullText).toBe(LONG_NAME);
    expect(label.querySelectorAll('span')).toHaveLength(width === '33px' ? 1 : 2);
    for (const span of label.querySelectorAll('span')) {
      expect(span.getAttribute('aria-hidden')).toBe('true');
    }
  });
});
