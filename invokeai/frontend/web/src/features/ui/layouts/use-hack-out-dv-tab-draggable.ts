import type { RefObject } from 'react';
import { useEffect } from 'react';

/**
 * Prevent undesired dnd behavior in Dockview tabs.
 *
 * Dockview always sets the draggable flag on its tab elements, even when dnd is disabled. This hook traverses
 * up from the provided ref to find the closest tab element and sets its `draggable` attribute to `false`.
 *
 * TODO: Remove this when https://github.com/mathuo/dockview/pull/961 is shipped.
 */
export const useHackOutDvTabDraggable = (ref: RefObject<HTMLElement>) => {
  useEffect(() => {
    const el = ref.current;
    if (!el) {
      return;
    }
    const parentTab = el.closest('.dv-tab');
    if (!parentTab) {
      return;
    }
    parentTab.setAttribute('draggable', 'false');

    const tabContainer = parentTab.closest('.dv-tabs-and-actions-container');
    if (!tabContainer) {
      return;
    }
    const voidContainer = tabContainer.querySelector('.dv-void-container');
    if (!voidContainer) {
      return;
    }
    voidContainer.setAttribute('draggable', 'false');
  }, [ref]);
};
