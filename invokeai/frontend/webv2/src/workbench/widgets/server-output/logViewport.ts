import { createExternalStoreCore } from '@platform/state/externalStoreCore';

/** Within this distance of the end the view counts as following new output. */
const FOLLOW_SLOP_PX = 4;

export interface LogViewportSnapshot {
  readonly following: boolean;
}

export interface LogViewport {
  getSnapshot(): LogViewportSnapshot;
  subscribe(listener: () => void): () => void;
  /** Ref callback for the scrolling element. */
  attachScroller(element: HTMLElement | null): void;
  /** Ref callback for the element holding `[data-block]` children of `[data-row]` rows. */
  attachContent(element: HTMLElement | null): void;
  jumpToLatest(): void;
}

interface Anchor {
  readonly rowId: string;
  /** How far the viewport's top sits below the row's top. */
  readonly offset: number;
}

/** First child whose bottom edge is below `top`, by binary search over children in display order. */
const firstChildBelow = (parent: Element, top: number): HTMLElement | null => {
  const children = parent.children;
  let low = 0;
  let high = children.length - 1;
  let found: HTMLElement | null = null;

  while (low <= high) {
    const middle = (low + high) >> 1;
    const child = children[middle] as HTMLElement;

    if (child.getBoundingClientRect().bottom > top) {
      found = child;
      high = middle - 1;
    } else {
      low = middle + 1;
    }
  }

  return found;
};

/**
 * Follows the end while the reader is there. Scrolled up, it holds the row at the top of the viewport in place
 * whatever rows are added, rewritten or evicted, by re-pinning that row after every DOM or size change: before
 * paint, so nothing visibly jumps. Browser scroll anchoring stays off on the scroller to leave one owner.
 */
export const createLogViewport = (): LogViewport => {
  const store = createExternalStoreCore<LogViewportSnapshot>({ following: true });
  let scroller: HTMLElement | null = null;
  let content: HTMLElement | null = null;
  let anchor: Anchor | null = null;
  let resizeObserver: ResizeObserver | null = null;
  let mutationObserver: MutationObserver | null = null;
  /** Sizes the last reconcile laid out for; a scroll under other sizes is the browser clamping, not the reader. */
  let knownSize = { client: -1, scroll: -1 };

  const measure = (): void => {
    if (!scroller || !content) {
      return;
    }

    const following = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight <= FOLLOW_SLOP_PX;

    anchor = null;
    if (!following) {
      const top = scroller.getBoundingClientRect().top;
      const block = firstChildBelow(content, top);
      const row = block ? firstChildBelow(block, top) : null;

      if (row?.dataset.row !== undefined) {
        anchor = { offset: top - row.getBoundingClientRect().top, rowId: row.dataset.row };
      }
    }

    store.patchSnapshot({ following });
  };

  const reconcile = (): void => {
    if (!scroller || !content) {
      return;
    }

    knownSize = { client: scroller.clientHeight, scroll: scroller.scrollHeight };

    if (store.getSnapshot().following) {
      scroller.scrollTop = scroller.scrollHeight;
      return;
    }

    const row = anchor ? content.querySelector<HTMLElement>(`[data-row="${anchor.rowId}"]`) : null;

    if (row && anchor) {
      const shift = row.getBoundingClientRect().top - scroller.getBoundingClientRect().top + anchor.offset;

      if (Math.abs(shift) >= 0.5) {
        scroller.scrollTop += shift;
      }
    }
  };

  const disconnect = (): void => {
    resizeObserver?.disconnect();
    mutationObserver?.disconnect();
    resizeObserver = null;
    mutationObserver = null;
  };

  const connect = (): void => {
    disconnect();
    if (!scroller || !content) {
      return;
    }

    resizeObserver = new ResizeObserver(reconcile);
    resizeObserver.observe(scroller);
    resizeObserver.observe(content);
    mutationObserver = new MutationObserver(reconcile);
    mutationObserver.observe(content, { characterData: true, childList: true, subtree: true });
    reconcile();
  };

  // Scroll events run before resize observers in a frame: one caused by content shrinking under the reader (wrapping
  // turned off, rows evicted) would read as reaching the end. Reconcile re-pins first; its own scroll then measures.
  const onScroll = (): void => {
    if (scroller && (scroller.clientHeight !== knownSize.client || scroller.scrollHeight !== knownSize.scroll)) {
      return;
    }
    measure();
  };

  return {
    attachContent: (element) => {
      content = element;
      connect();
    },
    attachScroller: (element) => {
      scroller?.removeEventListener('scroll', onScroll);
      scroller = element;
      scroller?.addEventListener('scroll', onScroll, { passive: true });
      connect();
    },
    getSnapshot: store.getSnapshot,
    jumpToLatest: () => {
      anchor = null;
      store.patchSnapshot({ following: true });
      reconcile();
    },
    subscribe: store.subscribe,
  };
};
