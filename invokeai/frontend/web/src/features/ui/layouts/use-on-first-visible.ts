import type { GridviewApi } from 'dockview';
import type { Atom } from 'nanostores';
import type { RefObject } from 'react';
import { useCallback, useEffect } from 'react';

import { MAIN_PANEL_ID } from './shared';

export const useOnFirstVisible = (elementRef: RefObject<HTMLElement>, callback: () => void): void => {
  useEffect(() => {
    const element = elementRef.current;
    if (!element) {
      return;
    }

    // Find the parent element that has display: none
    const findParentWithDisplay = (el: HTMLElement): HTMLElement | null => {
      let parent = el.parentElement;
      while (parent) {
        const computedStyle = window.getComputedStyle(parent);
        if (computedStyle.display === 'none') {
          return parent;
        }
        parent = parent.parentElement;
      }
      return null;
    };

    const targetParent = findParentWithDisplay(element);
    if (!targetParent) {
      return;
    }

    const observerCallback = () => {
      if (window.getComputedStyle(targetParent).display === 'none') {
        return;
      }
      observer.disconnect();
      callback();
    };
    const observer = new MutationObserver(observerCallback);

    observer.observe(targetParent, {
      attributes: true,
      attributeFilter: ['hidden', 'style', 'class'],
    });

    observerCallback();
    return () => {
      observer.disconnect();
    };
  }, [elementRef, callback]);
};

export const useResizeMainPanelOnFirstVisit = ($api: Atom<GridviewApi | null>, ref: RefObject<HTMLElement>) => {
  const resizeMainPanelOnFirstVisible = useCallback(() => {
    const api = $api.get();
    if (!api) {
      return;
    }
    const mainPanel = api.getPanel(MAIN_PANEL_ID);
    if (!mainPanel) {
      return;
    }
    if (mainPanel.width !== 0) {
      return;
    }
    let count = 0;
    const setSize = () => {
      if (count++ > 50) {
        return;
      }
      mainPanel.api.setSize({ width: Number.MAX_SAFE_INTEGER });
      if (mainPanel.width === 0) {
        requestAnimationFrame(setSize);
        return;
      }
    };
    setSize();
  }, [$api]);
  useOnFirstVisible(ref, resizeMainPanelOnFirstVisible);
};
