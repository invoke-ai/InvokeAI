/**
 * Calculate how many images fit in a row based on the current grid layout.
 *
 * TODO(psyche): We only need to do this when the gallery width changes, or when the galleryImageMinimumWidth value
 * changes. Cache this calculation.
 */
export const getItemsPerRow = (rootEl: HTMLDivElement): number => {
  // Start from root and find virtuoso grid elements
  const gridElement = rootEl.querySelector('.virtuoso-grid-list');

  if (!gridElement) {
    return 0;
  }

  const firstGridItem = gridElement.querySelector('.virtuoso-grid-item');

  if (!firstGridItem) {
    return 0;
  }

  const itemRect = firstGridItem.getBoundingClientRect();
  const containerRect = gridElement.getBoundingClientRect();

  // Get the computed gap from CSS
  const gridStyle = window.getComputedStyle(gridElement);
  const gapValue = gridStyle.gap;
  const gap = parseFloat(gapValue);

  if (isNaN(gap) || !itemRect.width || !itemRect.height || !containerRect.width || !containerRect.height) {
    return 0;
  }

  /**
   * You might be tempted to just do some simple math like:
   * const itemsPerRow = Math.floor(containerRect.width / itemRect.width);
   *
   * But floating point precision can cause issues with this approach, causing it to be off by 1 in some cases.
   *
   * Instead, we use a more robust approach that iteratively calculates how many items fit in the row.
   */
  let itemsPerRow = 0;
  let spaceUsed = 0;

  // Floating point precision can cause imagesPerRow to be 1 too small. Adding 1px to the container size fixes
  // this, without the possibility of accidentally adding an extra column.
  while (spaceUsed + itemRect.width <= containerRect.width + 1) {
    itemsPerRow++; // Increment the number of items
    spaceUsed += itemRect.width; // Add image size to the used space
    if (spaceUsed + gap <= containerRect.width) {
      spaceUsed += gap; // Add gap size to the used space after each item except after the last item
    }
  }

  return Math.max(1, itemsPerRow);
};
