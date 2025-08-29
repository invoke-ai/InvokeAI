import { useOverlayScrollbars } from 'overlayscrollbars-react';
import type { RefObject } from 'react';
import { useEffect, useState } from 'react';

/**
 * Handles the initialization of the overlay scrollbars for the queue list, returning the ref to the scroller element.
 */
export const useScrollableQueueList = (rootRef: RefObject<HTMLDivElement>) => {
  const [scroller, scrollerRef] = useState<HTMLElement | null>(null);
  const [initialize, osInstance] = useOverlayScrollbars({
    defer: true,
    events: {
      initialized(osInstance) {
        // force overflow styles
        const { viewport } = osInstance.elements();
        viewport.style.overflowX = `var(--os-viewport-overflow-x)`;
        viewport.style.overflowY = `var(--os-viewport-overflow-y)`;
      },
    },
    options: {
      scrollbars: {
        visibility: 'auto',
        autoHide: 'scroll',
        autoHideDelay: 1300,
        theme: 'os-theme-dark',
      },
    },
  });

  useEffect(() => {
    const { current: root } = rootRef;

    if (scroller && root) {
      initialize({
        target: root,
        elements: {
          viewport: scroller,
        },
      });
    }

    return () => {
      osInstance()?.destroy();
    };
  }, [scroller, initialize, osInstance, rootRef]);

  return scrollerRef;
};
