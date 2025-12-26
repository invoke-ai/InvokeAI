import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { clamp } from 'es-toolkit/compat';
import { useGalleryImageNames } from 'features/gallery/components/use-gallery-image-names';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { useCallback, useMemo } from 'react';

const SWIPE_THRESHOLD = 50; // Minimum distance in pixels to trigger swipe

export const useSwipeNavigation = () => {
  const dispatch = useAppDispatch();
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const { imageNames } = useGalleryImageNames();

  const isOnFirstItem = useMemo(
    () => (lastSelectedItem ? imageNames.at(0) === lastSelectedItem : false),
    [imageNames, lastSelectedItem]
  );
  const isOnLastItem = useMemo(
    () => (lastSelectedItem ? imageNames.at(-1) === lastSelectedItem : false),
    [imageNames, lastSelectedItem]
  );

  const navigateToPrevious = useCallback(() => {
    if (isOnFirstItem) {
      return;
    }
    const targetIndex = lastSelectedItem ? imageNames.findIndex((n) => n === lastSelectedItem) - 1 : 0;
    const clampedIndex = clamp(targetIndex, 0, imageNames.length - 1);
    const n = imageNames.at(clampedIndex);
    if (!n) {
      return;
    }
    dispatch(imageSelected(n));
  }, [dispatch, imageNames, lastSelectedItem, isOnFirstItem]);

  const navigateToNext = useCallback(() => {
    if (isOnLastItem) {
      return;
    }
    const targetIndex = lastSelectedItem ? imageNames.findIndex((n) => n === lastSelectedItem) + 1 : 0;
    const clampedIndex = clamp(targetIndex, 0, imageNames.length - 1);
    const n = imageNames.at(clampedIndex);
    if (!n) {
      return;
    }
    dispatch(imageSelected(n));
  }, [dispatch, imageNames, lastSelectedItem, isOnLastItem]);

  const onDragEnd = useCallback(
    (_event: MouseEvent | TouchEvent | PointerEvent, info: { offset: { x: number; y: number } }) => {
      // Swipe right (positive x) = go to previous image
      // Swipe left (negative x) = go to next image
      if (info.offset.x > SWIPE_THRESHOLD) {
        navigateToPrevious();
      } else if (info.offset.x < -SWIPE_THRESHOLD) {
        navigateToNext();
      }
    },
    [navigateToPrevious, navigateToNext]
  );

  return {
    onDragEnd,
    canNavigatePrevious: !isOnFirstItem,
    canNavigateNext: !isOnLastItem,
  };
};
