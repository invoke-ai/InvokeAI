import type { ListRange } from 'react-virtuoso';

/**
 * Gets the alignment for react-virtuoso's scrollToIndex function.
 * @param index The index of the item.
 * @param range The range of items currently visible.
 * @returns
 */
export const getScrollToIndexAlign = (index: number, range: ListRange): 'start' | 'end' => {
  if (index > (range.endIndex - range.startIndex) / 2 + range.startIndex) {
    return 'end';
  }
  return 'start';
};
