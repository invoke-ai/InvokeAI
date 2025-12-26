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
  const { imageNames, isFetching } = useGalleryImageNames();

  const currentIndex = useMemo(
    () => (lastSelectedItem ? imageNames.findIndex((n) => n === lastSelectedItem) : -1),
    [imageNames, lastSelectedItem]
  );

  const isOnFirstItem = useMemo(() => currentIndex === 0, [currentIndex]);
  const isOnLastItem = useMemo(() => currentIndex === imageNames.length - 1, [currentIndex, imageNames.length]);

  const previousImageName = useMemo(
    () => (currentIndex > 0 ? imageNames[currentIndex - 1] : null),
    [currentIndex, imageNames]
  );

  const nextImageName = useMemo(
    () => (currentIndex < imageNames.length - 1 ? imageNames[currentIndex + 1] : null),
    [currentIndex, imageNames]
  );

  const navigateToPrevious = useCallback(() => {
    if (isOnFirstItem || isFetching) {
      return;
    }
    const targetIndex = currentIndex - 1;
    const clampedIndex = clamp(targetIndex, 0, imageNames.length - 1);
    const n = imageNames.at(clampedIndex);
    if (!n) {
      return;
    }
    dispatch(imageSelected(n));
  }, [dispatch, imageNames, currentIndex, isOnFirstItem, isFetching]);

  const navigateToNext = useCallback(() => {
    if (isOnLastItem || isFetching) {
      return;
    }
    const targetIndex = currentIndex + 1;
    const clampedIndex = clamp(targetIndex, 0, imageNames.length - 1);
    const n = imageNames.at(clampedIndex);
    if (!n) {
      return;
    }
    dispatch(imageSelected(n));
  }, [dispatch, imageNames, currentIndex, isOnLastItem, isFetching]);

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
    previousImageName,
    nextImageName,
  };
};
