import type { RefCallback } from 'react';

/**
 * A row that scrolls sideways without a scrollbar: the wheel (`wheelScrollsHorizontally`), focus and the content cut off
 * at its edges carry it.
 */
export const HIDDEN_SCROLLBAR_CSS = { '&::-webkit-scrollbar': { display: 'none' }, scrollbarWidth: 'none' } as const;

/** A fractional zoom can leave a sub-pixel remainder at either end of a fully scrolled row; that hides nothing. */
export const SCROLL_EDGE_TOLERANCE_PX = 1;

export interface HiddenScrollEdges {
  end: boolean;
  start: boolean;
}

/** Which ends of a sideways scroller hide content. Right-to-left rows scroll toward negative offsets. */
export const measureHiddenEdges = (scroller: HTMLElement): HiddenScrollEdges => {
  const offset = Math.abs(scroller.scrollLeft);
  const range = scroller.scrollWidth - scroller.clientWidth;

  return { end: range - offset >= SCROLL_EDGE_TOLERANCE_PX, start: offset >= SCROLL_EDGE_TOLERANCE_PX };
};

/** How far the edge fade reaches into the row; an item revealed at a hiding edge should sit clear of it. */
export const SCROLL_EDGE_FADE_PX = 16;

const fade = (direction: 'left' | 'right', start: boolean, end: boolean) =>
  `linear-gradient(to ${direction}, ${start ? 'transparent' : 'black'}, black ${SCROLL_EDGE_FADE_PX}px, black calc(100% - ${SCROLL_EDGE_FADE_PX}px), ${end ? 'transparent' : 'black'})`;

const fadeCss = (start: boolean, end: boolean) => ({
  maskImage: fade('right', start, end),
  _rtl: { maskImage: fade('left', start, end) },
});

/** Fades the row out at each end `markHiddenEdges` marks as hiding content, whatever the row sits on. */
export const SCROLL_EDGE_FADE_CSS = {
  '&[data-hidden-start]': fadeCss(true, false),
  '&[data-hidden-end]': fadeCss(false, true),
  '&[data-hidden-start][data-hidden-end]': fadeCss(true, true),
} as const;

/**
 * Ref callback marking the ends of a sideways scroller that hide content as `data-hidden-start`/`data-hidden-end`.
 * Re-measured on scroll and whenever the scroller or one of its items resizes: an item can grow without resizing the
 * scroller (a count gaining a digit).
 */
export const markHiddenEdges: RefCallback<HTMLElement> = (element) => {
  if (!element) {
    return;
  }
  const update = () => {
    const { end, start } = measureHiddenEdges(element);
    element.toggleAttribute('data-hidden-start', start);
    element.toggleAttribute('data-hidden-end', end);
  };
  const resizes = new ResizeObserver(update);
  resizes.observe(element);
  for (const item of element.children) {
    resizes.observe(item);
  }
  const mutations = new MutationObserver((records) => {
    for (const record of records) {
      record.removedNodes.forEach((node) => node instanceof Element && resizes.unobserve(node));
      record.addedNodes.forEach((node) => node instanceof Element && resizes.observe(node));
    }
    update();
  });
  mutations.observe(element, { childList: true });
  element.addEventListener('scroll', update, { passive: true });
  update();

  return () => {
    resizes.disconnect();
    mutations.disconnect();
    element.removeEventListener('scroll', update);
  };
};
