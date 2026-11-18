import type { RefObject } from 'react';

import { useMountEffect } from '@platform/react/useMountEffect';

/**
 * Zag can retain phantom overflow after zero-size/remounted content. Check live viewport nodes and dispatch one
 * no-op scroll to remeasure; listeners must tolerate it.
 */
export const useScrollAreaPhantomHeal = (viewportRef: RefObject<HTMLElement | null>): void => {
  useMountEffect(() => {
    // Read the ref each time; presence remounts can invalidate captured nodes.
    const hasPhantomScrollbar = () => {
      const viewport = viewportRef.current;
      // Use direct children so nested scrollbars cannot report our overflow state.
      const scrollbar = viewport?.parentElement?.querySelector(':scope > [data-part="scrollbar"]');

      if (!viewport || !scrollbar) {
        return false;
      }

      const phantomY = scrollbar.hasAttribute('data-overflow-y') && viewport.scrollHeight <= viewport.clientHeight;
      const phantomX = scrollbar.hasAttribute('data-overflow-x') && viewport.scrollWidth <= viewport.clientWidth;

      return Boolean(phantomY || phantomX);
    };

    let disposed = false;
    let observed: HTMLElement | null = null;
    const timers: ReturnType<typeof setTimeout>[] = [];
    const observer = new ResizeObserver(() => check());

    const check = () => {
      const viewport = viewportRef.current;

      if (!viewport) {
        return;
      }

      if (viewport !== observed) {
        if (observed) {
          observer.unobserve(observed);
        }

        observer.observe(viewport);
        observed = viewport;
      }

      if (hasPhantomScrollbar()) {
        viewport.dispatchEvent(new Event('scroll'));
      }
    };

    // Timed checks recover missed remounts and rebind the observer to the current node.
    for (const delay of [64, 250, 600]) {
      timers.push(setTimeout(check, delay));
    }

    // A microtask leaves React's effect flush, where a scroll listener calling flushSync is an error, and still lands
    // before the next paint after mount.
    queueMicrotask(() => {
      if (!disposed) {
        check();
      }
    });

    return () => {
      disposed = true;
      timers.forEach(clearTimeout);
      observer.disconnect();
    };
  });
};
