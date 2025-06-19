import type { RefObject } from 'react';
import { useEffect } from 'react';

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
