import type { GridviewApi } from 'dockview';
import type { RefObject } from 'react';
import { useCallback, useEffect } from 'react';

import { MAIN_PANEL_ID } from './shared';

// Find the parent element that has display: none
const findParentWithDisplayNone = (el: HTMLElement): HTMLElement | null => {
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

export const useOnFirstVisible = (elementRef: RefObject<HTMLElement>, callback: () => void): void => {
  useEffect(() => {
    const element = elementRef.current;
    if (!element) {
      return;
    }

    const targetParent = findParentWithDisplayNone(element);
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

export const useResizeMainPanelOnFirstVisit = (api: GridviewApi | null, ref: RefObject<HTMLElement>) => {
  const resizeMainPanelOnFirstVisible = useCallback(() => {
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
  }, [api]);
  useOnFirstVisible(ref, resizeMainPanelOnFirstVisible);
};
