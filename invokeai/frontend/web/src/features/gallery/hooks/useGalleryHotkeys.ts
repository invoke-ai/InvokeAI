import { useAppSelector } from 'app/store/storeHooks';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { useGalleryImages } from 'features/gallery/hooks/useGalleryImages';
import { useGalleryNavigation } from 'features/gallery/hooks/useGalleryNavigation';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';

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

  const {
    areMoreImagesAvailable,
    handleLoadMoreImages,
    queryResult: { isFetching },
  } = useGalleryImages();

  const { handleLeftImage, handleRightImage, handleUpImage, handleDownImage, isOnLastImage, areImagesBelowCurrent } =
    useGalleryNavigation();

  useHotkeys(
    ['left', 'alt+left'],
    (e) => {
      canNavigateGallery && handleLeftImage(e.altKey);
    },
    [handleLeftImage, canNavigateGallery]
  );

  useHotkeys(
    ['right', 'alt+right'],
    (e) => {
      if (!canNavigateGallery) {
        return;
      }
      if (isOnLastImage && areMoreImagesAvailable && !isFetching) {
        handleLoadMoreImages();
        return;
      }
      if (!isOnLastImage) {
        handleRightImage(e.altKey);
      }
    },
    [isOnLastImage, areMoreImagesAvailable, handleLoadMoreImages, isFetching, handleRightImage, canNavigateGallery]
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
      if (!areImagesBelowCurrent && areMoreImagesAvailable && !isFetching) {
        handleLoadMoreImages();
        return;
      }
      handleDownImage(e.altKey);
    },
    { preventDefault: true },
    [areImagesBelowCurrent, areMoreImagesAvailable, handleLoadMoreImages, isFetching, handleDownImage]
  );
};
