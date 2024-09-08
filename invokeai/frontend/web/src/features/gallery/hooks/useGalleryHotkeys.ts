import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { $activeScopes } from 'common/hooks/interactionScopes';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useGalleryNavigation } from 'features/gallery/hooks/useGalleryNavigation';
import { useGalleryPagination } from 'features/gallery/hooks/useGalleryPagination';
import { selectListImagesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { $isGalleryPanelOpen } from 'features/ui/store/uiSlice';
import { computed } from 'nanostores';
import { useHotkeys } from 'react-hotkeys-hook';
import { useListImagesQuery } from 'services/api/endpoints/images';

const $leftRightHotkeysEnabled = computed($activeScopes, (activeScopes) => {
  // The left and right hotkeys can be used when the gallery is focused and the canvas is not focused, OR when the image viewer is focused.
  return !activeScopes.has('canvas') || activeScopes.has('imageViewer');
});

const $upDownHotkeysEnabled = computed([$activeScopes, $isGalleryPanelOpen], (activeScopes, isGalleryPanelOpen) => {
  // The up and down hotkeys can be used when the gallery is focused and the canvas is not focused, and the gallery panel is open.
  return !activeScopes.has('canvas') && isGalleryPanelOpen;
});

/**
 * Registers gallery hotkeys. This hook is a singleton.
 */
export const useGalleryHotkeys = () => {
  useAssertSingleton('useGalleryHotkeys');
  const { goNext, goPrev, isNextEnabled, isPrevEnabled } = useGalleryPagination();
  const queryArgs = useAppSelector(selectListImagesQueryArgs);
  const queryResult = useListImagesQuery(queryArgs);
  const leftRightHotkeysEnabled = useStore($leftRightHotkeysEnabled);
  const upDownHotkeysEnabled = useStore($upDownHotkeysEnabled);

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
    { preventDefault: true, enabled: leftRightHotkeysEnabled },
    [handleLeftImage, isOnFirstImageOfView, goPrev, isPrevEnabled, queryResult.isFetching, leftRightHotkeysEnabled]
  );

  useHotkeys(
    ['right', 'alt+right'],
    (e) => {
      if (isOnLastImageOfView && isNextEnabled && !queryResult.isFetching) {
        goNext(e.altKey ? 'alt+arrow' : 'arrow');
        return;
      }
      if (!isOnLastImageOfView) {
        handleRightImage(e.altKey);
      }
    },
    { preventDefault: true, enabled: leftRightHotkeysEnabled },
    [isOnLastImageOfView, goNext, isNextEnabled, queryResult.isFetching, handleRightImage, leftRightHotkeysEnabled]
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
    { preventDefault: true, enabled: upDownHotkeysEnabled },
    [handleUpImage, isOnFirstRow, goPrev, isPrevEnabled, queryResult.isFetching, upDownHotkeysEnabled]
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
    { preventDefault: true, enabled: upDownHotkeysEnabled },
    [isOnLastRow, goNext, isNextEnabled, queryResult.isFetching, handleDownImage, upDownHotkeysEnabled]
  );
};
