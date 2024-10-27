import { useAppSelector } from 'app/store/storeHooks';
import { virtuosoGridRefs } from 'features/gallery/components/ImageGrid/types';
import { selectHasMultipleImagesSelected } from 'features/gallery/store/gallerySelectors';
import { getIsVisible } from 'features/gallery/util/getIsVisible';
import { getScrollToIndexAlign } from 'features/gallery/util/getScrollToIndexAlign';
import { useEffect } from 'react';

/**
 * Scrolls an image into view when it is selected. This is necessary because
 * the image grid is virtualized, so the image may not be visible when it is
 * selected.
 *
 * Also handles when an image is selected programmatically - for example, when
 * auto-switching the new gallery images.
 *
 * @param imageContainerRef The ref to the image container.
 * @param isSelected Whether the image is selected.
 * @param index The index of the image in the gallery.
 * @returns
 *
 * @knip-ignore
 */
export const useScrollIntoView = (imageContainerRef: HTMLElement | null, isSelected: boolean, index: number) => {
  const areMultiplesSelected = useAppSelector(selectHasMultipleImagesSelected);

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

    const itemRect = imageContainerRef?.getBoundingClientRect();
    const rootRect = root.getBoundingClientRect();

    if (!itemRect || !getIsVisible(itemRect, rootRect)) {
      virtuoso.scrollToIndex({
        index,
        align: getScrollToIndexAlign(index, range),
      });
    }
  }, [isSelected, index, areMultiplesSelected, imageContainerRef]);
};
