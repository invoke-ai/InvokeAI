import { VirtuosoGalleryContext } from 'features/gallery/components/ImageGrid/types';
import { getScrollToIndexAlign } from 'features/gallery/util/getScrollToIndexAlign';
import { useEffect, useRef } from 'react';

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

    const itemRect = imageContainerRef.current.getBoundingClientRect();
    const rootRect = virtuosoContext.rootRef.current.getBoundingClientRect();
    const itemIsVisible =
      itemRect.top >= rootRect.top &&
      itemRect.bottom <= rootRect.bottom &&
      itemRect.left >= rootRect.left &&
      itemRect.right <= rootRect.right;

    if (!itemIsVisible) {
      virtuosoContext.virtuosoRef.current.scrollToIndex({
        index,
        behavior: 'smooth',
        align: getScrollToIndexAlign(
          index,
          virtuosoContext.virtuosoRangeRef.current
        ),
      });
    }
  }, [isSelected, index, selectionCount, virtuosoContext]);

  return imageContainerRef;
};
