import { useLayoutEffect, useRef, type RefObject } from 'react';

/**
 * Track offsets during scrolling: hidden-subtree cleanup sees zero after display:none. Restore before paint on
 * reveal; real unmount discards the instance's offsets.
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
