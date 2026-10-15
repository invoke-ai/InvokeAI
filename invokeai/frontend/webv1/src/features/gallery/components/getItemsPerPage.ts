import { getItemsPerRow } from './getItemsPerRow';

/**
 * Calculate how many images fit in the visible gallery area.
 */
export const getItemsPerPage = (rootEl: HTMLDivElement): number => {
  const gridElement = rootEl.querySelector('.virtuoso-grid-list');

  if (!gridElement) {
    return 0;
  }

  const firstGridItem = gridElement.querySelector('.virtuoso-grid-item');

  if (!firstGridItem) {
    return 0;
  }

  const itemRect = firstGridItem.getBoundingClientRect();
  const containerRect = rootEl.getBoundingClientRect();
  const gridStyle = window.getComputedStyle(gridElement);
  const gap = parseFloat(gridStyle.gap);

  if (isNaN(gap) || !itemRect.width || !itemRect.height || !containerRect.width || !containerRect.height) {
    return 0;
  }

  const itemsPerRow = getItemsPerRow(rootEl);

  if (itemsPerRow === 0) {
    return 0;
  }

  let rowsPerPage = 0;
  let spaceUsed = 0;

  while (spaceUsed + itemRect.height <= containerRect.height + 1) {
    rowsPerPage++;
    spaceUsed += itemRect.height;
    if (spaceUsed + gap <= containerRect.height) {
      spaceUsed += gap;
    }
  }

  return Math.max(1, rowsPerPage) * itemsPerRow;
};
