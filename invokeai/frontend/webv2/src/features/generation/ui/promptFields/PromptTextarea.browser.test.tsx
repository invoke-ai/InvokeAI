import { ChakraProvider } from '@chakra-ui/react';
import { getTextareaCaretRect } from '@features/generation/ui/promptFields/promptCaret';
import { PromptTextarea } from '@features/generation/ui/promptFields/PromptTextarea';
import {
  findHalfPixelWrapEdge,
  type HalfPixelWrapSide,
} from '@features/generation/ui/promptFields/textareaWrapWidth.testing';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

/**
 * The highlight mirror and the autocomplete caret must wrap exactly where the textarea does. `clientWidth` rounds to
 * whole pixels, so at a padding-box width ending in .5 a mirror sized from it wraps a word the textarea keeps, or the
 * reverse, and the caret lands a word and a line away from its glyph.
 */

const WORDS = 'masterpiece portrait of a lighthouse keeper at dusk with soft rim light and heavy fog'.split(' ');
// A fractional size gives the monospace advance a fraction, so word prefixes end at varied sub-pixel widths.
const FONT_SIZE = '13.3px';
const noop = () => undefined;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const settleLayout = () =>
  act(async () => {
    await document.fonts.ready;
    await new Promise((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(resolve));
    });
  });

const render = async (value: string, widthPx: number) => {
  if (!host) {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  }
  host.style.width = `${widthPx}px`;

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        {/* A short box makes the text overflow, so scrollHeight counts the textarea's own lines. */}
        <PromptTextarea
          aria-label="Prompt"
          defaultHeightPx={20}
          fontSize={FONT_SIZE}
          minHeightPx={20}
          resizeHandleAriaLabel="Resize"
          showSyntaxHighlighting
          value={value}
          onChange={noop}
        />
      </ChakraProvider>
    );
  });
  await settleLayout();
};

const textarea = () => host!.querySelector('textarea')!;
const mirror = () => host!.querySelector('pre')!;

const px = (value: string) => Number.parseFloat(value);
const lineHeightPx = () => px(getComputedStyle(textarea()).lineHeight);
const linesIn = (contentHeight: number) => Math.round(contentHeight / lineHeightPx());

const textareaLines = () => {
  const computed = getComputedStyle(textarea());

  return linesIn(textarea().scrollHeight - px(computed.paddingTop) - px(computed.paddingBottom));
};

const mirrorLines = () => {
  const computed = getComputedStyle(mirror());

  return linesIn(mirror().getBoundingClientRect().height - px(computed.paddingTop) - px(computed.paddingBottom));
};

/** The textarea's padding box, from its fractional border box. */
const paddingBoxWidth = () => {
  const computed = getComputedStyle(textarea());

  return textarea().getBoundingClientRect().width - px(computed.borderLeftWidth) - px(computed.borderRightWidth);
};

/** Renders the prefix of WORDS whose wrap edge sits half a pixel from a .5 padding-box width, at that width. */
const renderAtHalfPixelEdge = async (side: HalfPixelWrapSide) => {
  await render(WORDS.join(' '), 600);
  const edge = findHalfPixelWrapEdge(textarea(), WORDS, side);
  const hostOverhead = host!.getBoundingClientRect().width - paddingBoxWidth();

  await render(edge.text, edge.paddingBoxWidth + hostOverhead);
  expect(paddingBoxWidth()).toBeCloseTo(edge.paddingBoxWidth, 2);

  return edge;
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('prompt mirrors at a fractional width', () => {
  it('sizes the highlight mirror to the textarea wrap box, not its rounded clientWidth', async () => {
    await render('a lighthouse', 368.5);

    expect(paddingBoxWidth() % 1).toBeCloseTo(0.5, 2);
    expect(mirror().getBoundingClientRect().width).toBeCloseTo(paddingBoxWidth(), 2);
  });

  it.each([
    ['wraps', 2],
    ['fits', 1],
  ] as const)('wraps the highlight mirror like the textarea when the last word %s', async (side, lines) => {
    await renderAtHalfPixelEdge(side);

    expect(textareaLines()).toBe(lines);
    expect(mirrorLines()).toBe(lines);
  });

  it.each([
    ['wraps', 1],
    ['fits', 0],
  ] as const)('puts the caret on the line the textarea wraps to when the last word %s', async (side, line) => {
    const edge = await renderAtHalfPixelEdge(side);
    const computed = getComputedStyle(textarea());
    const contentTop =
      textarea().getBoundingClientRect().top +
      px(computed.borderTopWidth) +
      px(computed.paddingTop) -
      textarea().scrollTop;
    const caret = getTextareaCaretRect(textarea(), edge.edgeWordIndex);

    expect(textareaLines()).toBe(line + 1);
    expect(caret).not.toBeNull();
    expect(Math.round((caret!.y - contentTop) / lineHeightPx())).toBe(line);
  });
});
