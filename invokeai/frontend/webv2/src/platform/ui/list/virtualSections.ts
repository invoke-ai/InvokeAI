import { defaultRangeExtractor, type Range } from 'react-hook-tanstack-virtual';

/**
 * Range rules shared by virtualized lists whose section headers pin to the top. Headers and rows share one flat
 * index space; the header pinned at the top is the last one at or before the first visible index.
 */

export const lastHeaderAtOrBefore = (headerIndexes: readonly number[], index: number): number | null => {
  let low = 0;
  let high = headerIndexes.length - 1;
  let found: number | null = null;

  while (low <= high) {
    const middle = (low + high) >> 1;

    if (headerIndexes[middle]! <= index) {
      found = headerIndexes[middle]!;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }

  return found;
};

/**
 * The visible range plus the rows that must stay mounted wherever the list scrolls: the pinned header, and the
 * keyboard target (a focused row, or the option a combobox names as its active descendant). -1 keeps nothing.
 */
export const extractRangeKeepingPinned = (
  range: Range,
  headerIndexes: readonly number[],
  keptIndex: number
): number[] => {
  const indexes = defaultRangeExtractor(range);
  let changed = false;

  for (const index of [lastHeaderAtOrBefore(headerIndexes, range.startIndex), keptIndex]) {
    if (index !== null && index >= 0 && !indexes.includes(index)) {
      indexes.push(index);
      changed = true;
    }
  }

  return changed ? indexes.sort((a, b) => a - b) : indexes;
};

/** How far the next header has pushed the pinned one up; it can only push while it is mounted near the top. */
export const getPinnedHeaderShift = (
  nextHeaderStart: number | undefined,
  scrollOffset: number,
  pinnedHeight: number
): number => (nextHeaderStart === undefined ? 0 : Math.min(0, nextHeaderStart - scrollOffset - pinnedHeight));
