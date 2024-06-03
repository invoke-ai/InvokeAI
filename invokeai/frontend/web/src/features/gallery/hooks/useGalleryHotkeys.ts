import { useAppSelector } from 'app/store/storeHooks';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { useGalleryPagination } from 'features/gallery/hooks/useGalleryImages';
import { selectListImages2QueryArgs } from 'features/gallery/store/gallerySelectors';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useListImages2Query } from 'services/api/endpoints/images';

import { useGalleryNavigation } from './useGalleryNavigation';

/**
 * Registers gallery hotkeys. This hook is a singleton.
 */
export const useGalleryHotkeys = () => {
  const activeTabName = useAppSelector(activeTabNameSelector);
  const isStaging = useAppSelector(isStagingSelector);
  // block navigation on Unified Canvas tab when staging new images
  const canNavigateGallery = useMemo(() => {
    return activeTabName !== 'canvas' || !isStaging;
  }, [activeTabName, isStaging]);

  const { next, prev, isNextEnabled, isPrevEnabled } = useGalleryPagination();
  const queryArgs = useAppSelector(selectListImages2QueryArgs);
  const queryResult = useListImages2Query(queryArgs);

  const {
    handleLeftImage,
    handleRightImage,
    handleUpImage,
    handleDownImage,
    areImagesBelowCurrent,
    isOnFirstImageOfView,
    isOnLastImageOfView,
  } = useGalleryNavigation();

  useHotkeys(
    ['left', 'alt+left'],
    (e) => {
      if (isOnFirstImageOfView && isPrevEnabled && !queryResult.isFetching) {
        prev();
        return;
      }
      canNavigateGallery && handleLeftImage(e.altKey);
    },
    [handleLeftImage, canNavigateGallery, isOnFirstImageOfView]
  );

  useHotkeys(
    ['right', 'alt+right'],
    (e) => {
      if (!canNavigateGallery) {
        return;
      }
      if (isOnLastImageOfView && isNextEnabled && !queryResult.isFetching) {
        next();
        return;
      }
      if (!isOnLastImageOfView) {
        handleRightImage(e.altKey);
      }
    },
    [isOnLastImageOfView, next, isNextEnabled, queryResult.isFetching, handleRightImage, canNavigateGallery]
  );

  useHotkeys(
    ['up', 'alt+up'],
    (e) => {
      handleUpImage(e.altKey);
    },
    { preventDefault: true },
    [handleUpImage]
  );

  useHotkeys(
    ['down', 'alt+down'],
    (e) => {
      if (!areImagesBelowCurrent && isNextEnabled && !queryResult.isFetching) {
        next();
        return;
      }
      handleDownImage(e.altKey);
    },
    { preventDefault: true },
    [areImagesBelowCurrent, next, isNextEnabled, queryResult.isFetching, handleDownImage]
  );
};
