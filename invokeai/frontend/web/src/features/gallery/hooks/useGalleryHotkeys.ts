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
    return activeTabName !== 'unifiedCanvas' || !isStaging;
  }, [activeTabName, isStaging]);

  const {
    areMoreImagesAvailable,
    handleLoadMoreImages,
    queryResult: { isFetching },
  } = useGalleryImages();

  const { handleLeftImage, handleRightImage, handleUpImage, handleDownImage, isOnLastImage, areImagesBelowCurrent } =
    useGalleryNavigation();

  useHotkeys(
    'left',
    () => {
      canNavigateGallery && handleLeftImage();
    },
    [handleLeftImage, canNavigateGallery]
  );

  useHotkeys(
    'right',
    () => {
      if (!canNavigateGallery) {
        return;
      }
      if (isOnLastImage && areMoreImagesAvailable && !isFetching) {
        handleLoadMoreImages();
        return;
      }
      if (!isOnLastImage) {
        handleRightImage();
      }
    },
    [isOnLastImage, areMoreImagesAvailable, handleLoadMoreImages, isFetching, handleRightImage, canNavigateGallery]
  );

  useHotkeys(
    'up',
    () => {
      handleUpImage();
    },
    { preventDefault: true },
    [handleUpImage]
  );

  useHotkeys(
    'down',
    () => {
      if (!areImagesBelowCurrent && areMoreImagesAvailable && !isFetching) {
        handleLoadMoreImages();
        return;
      }
      handleDownImage();
    },
    { preventDefault: true },
    [areImagesBelowCurrent, areMoreImagesAvailable, handleLoadMoreImages, isFetching, handleDownImage]
  );
};
