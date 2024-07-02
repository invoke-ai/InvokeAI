import { useAppSelector } from 'app/store/storeHooks';
import { useGalleryNavigation } from 'features/gallery/hooks/useGalleryNavigation';
import { useGalleryPagination } from 'features/gallery/hooks/useGalleryPagination';
import { selectListImagesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { useHotkeys } from 'react-hotkeys-hook';
import { useListImagesQuery } from 'services/api/endpoints/images';

/**
 * Registers gallery hotkeys. This hook is a singleton.
 */
export const useGalleryHotkeys = () => {
  const isStaging = useAppSelector((s) => s.canvasV2.stagingArea.isStaging);

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
      handleLeftImage(e.altKey);
    },
    [handleLeftImage, isOnFirstImageOfView, goPrev, isPrevEnabled, queryResult.isFetching]
  );

  useHotkeys(
    ['right', 'alt+right'],
    (e) => {
      if (isStaging) {
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
    [isStaging, isOnLastImageOfView, goNext, isNextEnabled, queryResult.isFetching, handleRightImage]
  );

  useHotkeys(
    ['up', 'alt+up'],
    (e) => {
      if (isStaging) {
        return;
      }
      if (isOnFirstRow && isPrevEnabled && !queryResult.isFetching) {
        goPrev(e.altKey ? 'alt+arrow' : 'arrow');
        return;
      }
      handleUpImage(e.altKey);
    },
    { preventDefault: true },
    [isStaging, handleUpImage, isOnFirstRow, goPrev, isPrevEnabled, queryResult.isFetching]
  );

  useHotkeys(
    ['down', 'alt+down'],
    (e) => {
      if (isStaging) {
        return;
      }
      if (isOnLastRow && isNextEnabled && !queryResult.isFetching) {
        goNext(e.altKey ? 'alt+arrow' : 'arrow');
        return;
      }
      handleDownImage(e.altKey);
    },
    { preventDefault: true },
    [isStaging, isOnLastRow, goNext, isNextEnabled, queryResult.isFetching, handleDownImage]
  );
};
