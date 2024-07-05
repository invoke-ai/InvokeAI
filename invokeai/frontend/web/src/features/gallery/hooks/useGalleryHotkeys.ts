import { useAppSelector } from 'app/store/storeHooks';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { useGalleryNavigation } from 'features/gallery/hooks/useGalleryNavigation';
import { useGalleryPagination } from 'features/gallery/hooks/useGalleryPagination';
import { selectListImagesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useListImagesQuery } from 'services/api/endpoints/images';

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

  const { goNext, goPrev, isNextEnabled, isPrevEnabled } = useGalleryPagination();
  const queryArgs = useAppSelector(selectListImagesQueryArgs);
  const queryResult = useListImagesQuery(queryArgs);

  const {
    handleLeftImage,
    handleRightImage,
    handleUpImage,
    handleDownImage,
    isOnFirstRow,
    isOnLastRow,
    isOnFirstImageOfView,
    isOnLastImageOfView,
  } = useGalleryNavigation();

  useHotkeys(
    ['left', 'alt+left'],
    (e) => {
      if (isOnFirstImageOfView && isPrevEnabled && !queryResult.isFetching) {
        goPrev(e.altKey ? 'alt+arrow' : 'arrow');
        return;
      }
      canNavigateGallery && handleLeftImage(e.altKey);
    },
    [handleLeftImage, canNavigateGallery, isOnFirstImageOfView, goPrev, isPrevEnabled, queryResult.isFetching]
  );

  useHotkeys(
    ['right', 'alt+right'],
    (e) => {
      if (!canNavigateGallery) {
        return;
      }
      if (isOnLastImageOfView && isNextEnabled && !queryResult.isFetching) {
        goNext(e.altKey ? 'alt+arrow' : 'arrow');
        return;
      }
      if (!isOnLastImageOfView) {
        handleRightImage(e.altKey);
      }
    },
    [isOnLastImageOfView, goNext, isNextEnabled, queryResult.isFetching, handleRightImage, canNavigateGallery]
  );

  useHotkeys(
    ['up', 'alt+up'],
    (e) => {
      if (isOnFirstRow && isPrevEnabled && !queryResult.isFetching) {
        goPrev(e.altKey ? 'alt+arrow' : 'arrow');
        return;
      }
      handleUpImage(e.altKey);
    },
    { preventDefault: true },
    [handleUpImage, canNavigateGallery, isOnFirstRow, goPrev, isPrevEnabled, queryResult.isFetching]
  );

  useHotkeys(
    ['down', 'alt+down'],
    (e) => {
      if (isOnLastRow && isNextEnabled && !queryResult.isFetching) {
        goNext(e.altKey ? 'alt+arrow' : 'arrow');
        return;
      }
      handleDownImage(e.altKey);
    },
    { preventDefault: true },
    [isOnLastRow, goNext, isNextEnabled, queryResult.isFetching, handleDownImage]
  );
};
