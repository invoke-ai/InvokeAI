import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { $activeScopes } from 'common/hooks/interactionScopes';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { $canvasRightPanelTab } from 'features/controlLayers/components/CanvasRightPanel';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import { useGalleryNavigation } from 'features/gallery/hooks/useGalleryNavigation';
import { useGalleryPagination } from 'features/gallery/hooks/useGalleryPagination';
import { selectListImagesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { $isRightPanelOpen } from 'features/ui/store/uiSlice';
import { computed } from 'nanostores';
import { useCallback, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useListImagesQuery } from 'services/api/endpoints/images';

const $leftRightHotkeysEnabled = computed($activeScopes, (activeScopes) => {
  // The left and right hotkeys can be used when the gallery is focused and the canvas is not focused, OR when the image viewer is focused.
  return !activeScopes.has('canvas') || activeScopes.has('imageViewer');
});

const $upDownHotkeysEnabled = computed([$activeScopes, $isRightPanelOpen], (activeScopes, isGalleryPanelOpen) => {
  // The up and down hotkeys can be used when the gallery is focused and the canvas is not focused, and the gallery panel is open.
  return !activeScopes.has('canvas') && isGalleryPanelOpen;
});

/**
 * Registers gallery hotkeys. This hook is a singleton.
 */
export const useGalleryHotkeys = () => {
  useAssertSingleton('useGalleryHotkeys');
  const { goNext, goPrev, isNextEnabled, isPrevEnabled } = useGalleryPagination();
  const dispatch = useAppDispatch();
  const selection = useAppSelector((s) => s.gallery.selection);
  const queryArgs = useAppSelector(selectListImagesQueryArgs);
  const queryResult = useListImagesQuery(queryArgs);
  const leftRightHotkeysEnabled = useStore($leftRightHotkeysEnabled);
  const upDownHotkeysEnabled = useStore($upDownHotkeysEnabled);
  const canvasRightPanelTab = useStore($canvasRightPanelTab);
  const appTab = useAppSelector(selectActiveTab);

  // When we are on the canvas tab, we need to disable the delete hotkey when the user is focused on the layers tab in
  // the right hand panel, because the same hotkey is used to delete layers.
  const isDeleteEnabledByTab = useMemo(() => {
    if (appTab !== 'canvas') {
      return true;
    }
    return canvasRightPanelTab === 'gallery';
  }, [appTab, canvasRightPanelTab]);

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

  const handleDelete = useCallback(() => {
    if (!selection.length) {
      return;
    }
    dispatch(imagesToDeleteSelected(selection));
  }, [dispatch, selection]);

  useHotkeys(['delete', 'backspace'], handleDelete, { enabled: leftRightHotkeysEnabled && isDeleteEnabledByTab }, [
    leftRightHotkeysEnabled,
    isDeleteEnabledByTab,
  ]);
};
