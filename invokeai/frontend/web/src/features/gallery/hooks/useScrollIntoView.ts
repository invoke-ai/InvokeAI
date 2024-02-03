import { virtuosoGridRefs } from 'features/gallery/components/ImageGrid/types';
import { getIsVisible } from 'features/gallery/util/getIsVisible';
import { getScrollToIndexAlign } from 'features/gallery/util/getScrollToIndexAlign';
import { useEffect, useRef } from 'react';

/**
 * Scrolls an image into view when it is selected. This is necessary because
 * the image grid is virtualized, so the image may not be visible when it is
 * selected.
 *
 * Also handles when an image is selected programmatically - for example, when
 * auto-switching the new gallery images.
 *
 * @param isSelected Whether the image is selected.
 * @param index The index of the image in the gallery.
 * @param selectionCount The number of images selected.
 * @returns
 */
export const useScrollIntoView = (isSelected: boolean, index: number, areMultiplesSelected: boolean) => {
  const imageContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isSelected || areMultiplesSelected) {
      return;
    }

    const virtuosoContext = virtuosoGridRefs.get();
    const range = virtuosoContext.virtuosoRangeRef?.current;
    const root = virtuosoContext.rootRef?.current;
    const virtuoso = virtuosoContext.virtuosoRef?.current;

    if (!range || !virtuoso || !root) {
      return;
    }

    const itemRect = imageContainerRef.current?.getBoundingClientRect();
    const rootRect = root.getBoundingClientRect();

    if (!itemRect || !getIsVisible(itemRect, rootRect)) {
      virtuoso.scrollToIndex({
        index,
        align: getScrollToIndexAlign(index, range),
      });
    }
  }, [isSelected, index, areMultiplesSelected]);

  return imageContainerRef;
};
