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

  const isOnFirstItem = useMemo(
    () => (lastSelectedItem ? imageNames.at(0) === lastSelectedItem : false),
    [imageNames, lastSelectedItem]
  );
  const isOnLastItem = useMemo(
    () => (lastSelectedItem ? imageNames.at(-1) === lastSelectedItem : false),
    [imageNames, lastSelectedItem]
  );

  const goToPreviousImage = useCallback(() => {
    const targetIndex = lastSelectedItem ? imageNames.findIndex((n) => n === lastSelectedItem) - 1 : 0;
    const clampedIndex = clamp(targetIndex, 0, imageNames.length - 1);
    const imageName = imageNames.at(clampedIndex);
    if (!imageName) {
      return;
    }
    dispatch(imageSelected(imageName));
  }, [dispatch, imageNames, lastSelectedItem]);

  const goToNextImage = useCallback(() => {
    const targetIndex = lastSelectedItem ? imageNames.findIndex((n) => n === lastSelectedItem) + 1 : 0;
    const clampedIndex = clamp(targetIndex, 0, imageNames.length - 1);
    const imageName = imageNames.at(clampedIndex);
    if (!imageName) {
      return;
    }
    dispatch(imageSelected(imageName));
  }, [dispatch, imageNames, lastSelectedItem]);

  return { goToPreviousImage, goToNextImage, isOnFirstItem, isOnLastItem, isFetching };
};
