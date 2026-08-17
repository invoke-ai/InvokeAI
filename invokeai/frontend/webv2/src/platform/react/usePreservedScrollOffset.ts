import { useLayoutEffect, useRef, type RefObject } from 'react';

/**
 * Keeps a scroll container's offset across the container ceasing to be rendered.
 *
 * The shell keeps widgets mounted across layout switches and hides them with
 * `display: none`, which takes the element out of layout entirely: it has no
 * scrollable overflow, its `scrollTop` reads 0, and the browser's own scroll
 * restoration only survives if the scrollable content happens to be unchanged
 * when it comes back. A plain, non-virtualized list — the gallery's board
 * list is the real reproduction case — has its content sized from actual DOM
 * layout, which is not guaranteed to come back identical, and the offset is
 * simply lost. A virtualized list is not the problem: its content height is a
 * pure function of row count and estimated sizes rather than of being laid
 * out at all, so it survives the round trip unaided and needs no help here.
 *
 * The offset is tracked as the user scrolls rather than read back at teardown,
 * because by the time a hidden subtree's effects are cleaned up the element is
 * already out of layout and reports `scrollTop` as 0 — reading it there records
 * nothing. Restoring happens in a layout effect, which runs whenever the subtree
 * is shown again (a real remount and a keep-alive reveal are indistinguishable
 * here, and both want the same thing) and lands before paint, so the list never
 * shows a frame at the top.
 *
 * Scoped to one component instance by construction: a genuine unmount takes the
 * remembered offset with it, exactly as before.
 */
export const usePreservedScrollOffset = (ref: RefObject<HTMLElement | null>): void => {
  const offsetRef = useRef({ left: 0, top: 0 });

  useLayoutEffect(() => {
    const element = ref.current;

    if (!element) {
      return;
    }

    if (offsetRef.current.top > 0) {
      element.scrollTop = offsetRef.current.top;
    }

    if (offsetRef.current.left > 0) {
      element.scrollLeft = offsetRef.current.left;
    }

    const recordOffset = () => {
      offsetRef.current = { left: element.scrollLeft, top: element.scrollTop };
    };

    element.addEventListener('scroll', recordOffset, { passive: true });

    return () => element.removeEventListener('scroll', recordOffset);
  }, [ref]);
};
