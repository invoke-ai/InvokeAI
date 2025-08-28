import type { ListRange, VirtuosoGridHandle } from 'react-virtuoso';

import { log } from './VideoGallery';

/**
 * Scroll the item at the given index into view if it is not currently visible.
 */
export const scrollIntoView = (
  targetItemId: string,
  itemIds: string[],
  rootEl: HTMLDivElement,
  virtuosoGridHandle: VirtuosoGridHandle,
  range: ListRange
) => {
  if (range.endIndex === 0) {
    // No range is rendered; no need to scroll to anything.
    log.trace('Not scrolling into view: Range endIdex is 0');
    return;
  }

  const targetIndex = itemIds.findIndex((name) => name === targetItemId);

  if (targetIndex === -1) {
    // The image isn't in the currently rendered list.
    log.trace('Not scrolling into view: targetIndex is -1');
    return;
  }

  const targetItem = rootEl.querySelector(`.virtuoso-grid-item:has([data-item-id="${targetItemId}"])`) as HTMLElement;

  if (!targetItem) {
    if (targetIndex > range.endIndex) {
      log.trace(
        {
          index: targetIndex,
          behavior: 'auto',
          align: 'start',
        },
        'Scrolling into view: not in DOM'
      );
      virtuosoGridHandle.scrollToIndex({
        index: targetIndex,
        behavior: 'auto',
        align: 'start',
      });
    } else if (targetIndex < range.startIndex) {
      log.trace(
        {
          index: targetIndex,
          behavior: 'auto',
          align: 'end',
        },
        'Scrolling into view: not in DOM'
      );
      virtuosoGridHandle.scrollToIndex({
        index: targetIndex,
        behavior: 'auto',
        align: 'end',
      });
    } else {
      log.debug(
        `Unable to find image ${targetItemId} at index ${targetIndex} but it is in the rendered range ${range.startIndex}-${range.endIndex}`
      );
    }
    return;
  }

  // We found the image in the DOM, but it might be in the overscan range - rendered but not in the visible viewport.
  // Check if it is in the viewport and scroll if necessary.

  const itemRect = targetItem.getBoundingClientRect();
  const rootRect = rootEl.getBoundingClientRect();

  if (itemRect.top < rootRect.top) {
    log.trace(
      {
        index: targetIndex,
        behavior: 'auto',
        align: 'start',
      },
      'Scrolling into view: in overscan'
    );
    virtuosoGridHandle.scrollToIndex({
      index: targetIndex,
      behavior: 'auto',
      align: 'start',
    });
  } else if (itemRect.bottom > rootRect.bottom) {
    log.trace(
      {
        index: targetIndex,
        behavior: 'auto',
        align: 'end',
      },
      'Scrolling into view: in overscan'
    );
    virtuosoGridHandle.scrollToIndex({
      index: targetIndex,
      behavior: 'auto',
      align: 'end',
    });
  } else {
    // Image is already in view
    log.debug('Not scrolling into view: Image is already in view');
  }

  return;
};
