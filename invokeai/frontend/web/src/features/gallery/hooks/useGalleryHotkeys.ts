import { useGalleryImages } from 'features/gallery/hooks/useGalleryImages';
import { useGalleryNavigation } from 'features/gallery/hooks/useGalleryNavigation';
import { useHotkeys } from 'react-hotkeys-hook';

/**
 * Registers gallery hotkeys. This hook is a singleton.
 */
export const useGalleryHotkeys = () => {
  const {
    areMoreImagesAvailable,
    handleLoadMoreImages,
    queryResult: { isFetching },
  } = useGalleryImages();

  const {
    handleLeftImage,
    handleRightImage,
    handleUpImage,
    handleDownImage,
    isOnLastImage,
    areImagesBelowCurrent,
  } = useGalleryNavigation();

  useHotkeys(
    'left',
    () => {
      handleLeftImage();
    },
    [handleLeftImage]
  );

  useHotkeys(
    'right',
    () => {
      if (isOnLastImage && areMoreImagesAvailable && !isFetching) {
        handleLoadMoreImages();
        return;
      }
      if (!isOnLastImage) {
        handleRightImage();
      }
    },
    [
      isOnLastImage,
      areMoreImagesAvailable,
      handleLoadMoreImages,
      isFetching,
      handleRightImage,
    ]
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
    [
      areImagesBelowCurrent,
      areMoreImagesAvailable,
      handleLoadMoreImages,
      isFetching,
      handleDownImage,
    ]
  );
};
