import { useEffect, useRef } from 'react';
import { VirtuosoGalleryContext } from '../components/ImageGrid/types';

export const useScrollToVisible = (
  isSelected: boolean,
  index: number,
  selectionCount: number,
  virtuosoContext: VirtuosoGalleryContext
) => {
  const imageContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (
      !isSelected ||
      selectionCount !== 1 ||
      !virtuosoContext.rootRef.current ||
      !virtuosoContext.virtuosoRef.current ||
      !virtuosoContext.virtuosoRangeRef.current ||
      !imageContainerRef.current
    ) {
      return;
    }

    const root = virtuosoContext.rootRef.current;
    const virtuoso = virtuosoContext.virtuosoRef.current;
    const item = imageContainerRef.current;
    const range = virtuosoContext.virtuosoRangeRef.current;
    const itemRect = item.getBoundingClientRect();
    const rootRect = root.getBoundingClientRect();
    const itemIsVisible =
      itemRect.top >= rootRect.top &&
      itemRect.bottom <= rootRect.bottom &&
      itemRect.left >= rootRect.left &&
      itemRect.right <= rootRect.right;

    if (!itemIsVisible) {
      virtuoso.scrollToIndex({
        index,
        behavior: 'smooth',
        align:
          index > (range.endIndex - range.startIndex) / 2 + range.startIndex
            ? 'end'
            : 'start',
      });
    }
  }, [isSelected, index, selectionCount, virtuosoContext]);

  return imageContainerRef;
};
