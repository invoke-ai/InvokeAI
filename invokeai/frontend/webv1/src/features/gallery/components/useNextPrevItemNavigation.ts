import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { clamp } from 'es-toolkit/compat';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { useCallback, useMemo } from 'react';

import { useGalleryImageNames } from './use-gallery-image-names';

export const useNextPrevItemNavigation = () => {
  const dispatch = useAppDispatch();
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const { imageNames, isFetching } = useGalleryImageNames();

  const currentIndex = useMemo(
    () => (lastSelectedItem ? imageNames.findIndex((n) => n === lastSelectedItem) : -1),
    [imageNames, lastSelectedItem]
  );
  const isOnFirstItem = currentIndex === 0;
  const isOnLastItem = currentIndex >= 0 && currentIndex === imageNames.length - 1;

  const navigateBy = useCallback(
    (delta: number) => {
      const maxIndex = imageNames.length - 1;
      if (maxIndex < 0) {
        return;
      }

      const targetIndex = currentIndex >= 0 ? clamp(currentIndex + delta, 0, maxIndex) : 0;
      const imageName = imageNames[targetIndex];
      if (!imageName) {
        return;
      }
      dispatch(imageSelected(imageName));
    },
    [currentIndex, dispatch, imageNames]
  );

  const goToPreviousImage = useCallback(() => {
    navigateBy(-1);
  }, [navigateBy]);

  const goToNextImage = useCallback(() => {
    navigateBy(1);
  }, [navigateBy]);

  return { goToPreviousImage, goToNextImage, isOnFirstItem, isOnLastItem, isFetching };
};
