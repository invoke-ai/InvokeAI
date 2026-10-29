import { ChakraProvider } from '@chakra-ui/react';
import { PromptTextarea } from '@features/generation/ui/promptFields/PromptTextarea';
import {
  findHalfPixelWrapEdge,
  type HalfPixelWrapSide,
} from '@features/generation/ui/promptFields/textareaWrapWidth.testing';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (
  value: string,
  widthPx = 320,
  { fontSize, heightPx = 200 }: { fontSize?: string; heightPx?: number } = {}
) => {
  if (!host) {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  }
  host.style.width = `${widthPx}px`;

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <PromptTextarea
          aria-label="Values"
          defaultHeightPx={heightPx}
          fontSize={fontSize}
          minHeightPx={Math.min(96, heightPx)}
          readOnly
          resizeHandleAriaLabel="Resize"
          showLineNumbers
          showSyntaxHighlighting
          value={value}
        />
      </ChakraProvider>
    );
  });
  // The gutter sizes itself from a measured mirror, so let layout settle.
  await act(async () => {
    await document.fonts.ready;
    await new Promise((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(resolve));
    });
  });
};

const gutterEntries = () => [...host!.querySelectorAll('[data-line-number]')];

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('prompt line numbers', () => {
  it('indents the text clear of the gutter so numbers never overlap it', async () => {
    // Fractional Chakra spacing variables are invalid inside calc.
    await render('toyota+\nhonda\nacura--');

    const textarea = host!.querySelector('textarea')!;
    const gutter = host!.querySelector('[data-line-number]')!.parentElement!.parentElement!;
    const textStart = textarea.getBoundingClientRect().left + parseFloat(getComputedStyle(textarea).paddingInlineStart);

    expect(parseFloat(getComputedStyle(textarea).paddingInlineStart)).toBeGreaterThan(0);
    expect(gutter.getBoundingClientRect().right).toBeLessThanOrEqual(textStart);
  });

  it('numbers every logical line', async () => {
    await render('red\ngreen\nblue');

    expect(gutterEntries().map((entry) => entry.textContent)).toEqual(['1', '2', '3']);
  });

  it('numbers a blank line rather than skipping it', async () => {
    await render('red\n\nblue');

    expect(gutterEntries().map((entry) => entry.textContent)).toEqual(['1', '2', '3']);
  });

  it('gives a wrapped line one number and the height of its wrap', async () => {
    const long = 'a '.repeat(120).trim();

    await render(`short\n${long}\nshort`);

    const heights = gutterEntries().map((entry) => entry.getBoundingClientRect().height);

    expect(gutterEntries().map((entry) => entry.textContent)).toEqual(['1', '2', '3']);
    // Wrapped lines need taller gutter entries but only one logical line number.
    expect(heights[1]).toBeGreaterThan(heights[0] * 2);
    expect(heights[2]).toBeCloseTo(heights[0], 1);
  });

  it.each([
    ['wraps', 2],
    ['fits', 1],
  ] as const satisfies ReadonlyArray<readonly [HalfPixelWrapSide, number]>)(
    'sizes a line the way the textarea wraps it at a half-pixel width, when its last word %s',
    async (side, lines) => {
      const words = 'masterpiece portrait of a lighthouse keeper at dusk with soft rim light and heavy fog'.split(' ');
      // A fractional font size spreads word ends across sub-pixel widths; a short box lets scrollHeight count lines.
      const options = { fontSize: '13.3px', heightPx: 20 };
      const px = (value: string) => Number.parseFloat(value);
      const paddingBoxWidth = (textarea: HTMLTextAreaElement) =>
        textarea.getBoundingClientRect().width - px(getComputedStyle(textarea).borderLeftWidth) * 2;

      await render(words.join(' '), 600, options);
      const textarea = host!.querySelector('textarea')!;
      const edge = findHalfPixelWrapEdge(textarea, words, side);
      await render(
        edge.text,
        edge.paddingBoxWidth + host!.getBoundingClientRect().width - paddingBoxWidth(textarea),
        options
      );

      const computed = getComputedStyle(textarea);
      const lineHeight = px(computed.lineHeight);
      const textareaLines = Math.round(
        (textarea.scrollHeight - px(computed.paddingTop) - px(computed.paddingBottom)) / lineHeight
      );

      expect(paddingBoxWidth(textarea)).toBeCloseTo(edge.paddingBoxWidth, 2);
      expect(textareaLines).toBe(lines);
      expect(gutterEntries().map((entry) => Math.round(entry.getBoundingClientRect().height / lineHeight))).toEqual([
        lines,
      ]);
    }
  );
});
