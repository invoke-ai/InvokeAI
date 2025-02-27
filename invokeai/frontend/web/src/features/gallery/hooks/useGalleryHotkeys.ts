import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useIsRegionFocused } from 'common/hooks/focus';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import { useGalleryNavigation } from 'features/gallery/hooks/useGalleryNavigation';
import { useGalleryPagination } from 'features/gallery/hooks/useGalleryPagination';
import { selectListImagesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { selectActiveTab, selectActiveTabCanvasRightPanel } from 'features/ui/store/uiSelectors';
import { useMemo } from 'react';
import { useListImagesQuery } from 'services/api/endpoints/images';

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
  const canvasRightPanelTab = useAppSelector(selectActiveTabCanvasRightPanel);
  const appTab = useAppSelector(selectActiveTab);
  const isWorkflowsFocused = useIsRegionFocused('workflows');
  const isGalleryFocused = useIsRegionFocused('gallery');
  const isImageViewerFocused = useIsRegionFocused('viewer');

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

  useRegisteredHotkeys({
    id: 'galleryNavLeft',
    category: 'gallery',
    callback: () => {
      if (isOnFirstImageOfView && isPrevEnabled && !queryResult.isFetching) {
        goPrev('arrow');
        return;
      }
      handleLeftImage(false);
    },
    options: { preventDefault: true, enabled: isGalleryFocused || isImageViewerFocused },
    dependencies: [
      handleLeftImage,
      isOnFirstImageOfView,
      goPrev,
      isPrevEnabled,
      queryResult.isFetching,
      isGalleryFocused,
      isImageViewerFocused,
    ],
  });

  useRegisteredHotkeys({
    id: 'galleryNavRight',
    category: 'gallery',
    callback: () => {
      if (isOnLastImageOfView && isNextEnabled && !queryResult.isFetching) {
        goNext('arrow');
        return;
      }
      if (!isOnLastImageOfView) {
        handleRightImage(false);
      }
    },
    options: { preventDefault: true, enabled: isGalleryFocused || isImageViewerFocused },
    dependencies: [
      isOnLastImageOfView,
      goNext,
      isNextEnabled,
      queryResult.isFetching,
      handleRightImage,
      isGalleryFocused,
      isImageViewerFocused,
    ],
  });

  useRegisteredHotkeys({
    id: 'galleryNavUp',
    category: 'gallery',
    callback: () => {
      if (isOnFirstRow && isPrevEnabled && !queryResult.isFetching) {
        goPrev('arrow');
        return;
      }
      handleUpImage(false);
    },
    options: { preventDefault: true, enabled: isGalleryFocused },
    dependencies: [handleUpImage, isOnFirstRow, goPrev, isPrevEnabled, queryResult.isFetching, isGalleryFocused],
  });

  useRegisteredHotkeys({
    id: 'galleryNavDown',
    category: 'gallery',
    callback: () => {
      if (isOnLastRow && isNextEnabled && !queryResult.isFetching) {
        goNext('arrow');
        return;
      }
      handleDownImage(false);
    },
    options: { preventDefault: true, enabled: isGalleryFocused },
    dependencies: [isOnLastRow, goNext, isNextEnabled, queryResult.isFetching, handleDownImage, isGalleryFocused],
  });

  useRegisteredHotkeys({
    id: 'galleryNavLeftAlt',
    category: 'gallery',
    callback: () => {
      if (isOnFirstImageOfView && isPrevEnabled && !queryResult.isFetching) {
        goPrev('alt+arrow');
        return;
      }
      handleLeftImage(true);
    },
    options: { preventDefault: true, enabled: isGalleryFocused || isImageViewerFocused },
    dependencies: [
      handleLeftImage,
      isOnFirstImageOfView,
      goPrev,
      isPrevEnabled,
      queryResult.isFetching,
      isGalleryFocused,
      isImageViewerFocused,
    ],
  });

  useRegisteredHotkeys({
    id: 'galleryNavRightAlt',
    category: 'gallery',
    callback: () => {
      if (isOnLastImageOfView && isNextEnabled && !queryResult.isFetching) {
        goNext('alt+arrow');
        return;
      }
      if (!isOnLastImageOfView) {
        handleRightImage(true);
      }
    },
    options: { preventDefault: true, enabled: isGalleryFocused || isImageViewerFocused },
    dependencies: [
      isOnLastImageOfView,
      goNext,
      isNextEnabled,
      queryResult.isFetching,
      handleRightImage,
      isGalleryFocused,
      isImageViewerFocused,
    ],
  });

  useRegisteredHotkeys({
    id: 'galleryNavUpAlt',
    category: 'gallery',
    callback: () => {
      if (isOnFirstRow && isPrevEnabled && !queryResult.isFetching) {
        goPrev('alt+arrow');
        return;
      }
      handleUpImage(true);
    },
    options: { preventDefault: true, enabled: isGalleryFocused },
    dependencies: [handleUpImage, isOnFirstRow, goPrev, isPrevEnabled, queryResult.isFetching, isGalleryFocused],
  });

  useRegisteredHotkeys({
    id: 'galleryNavDownAlt',
    category: 'gallery',
    callback: () => {
      if (isOnLastRow && isNextEnabled && !queryResult.isFetching) {
        goNext('alt+arrow');
        return;
      }
      handleDownImage(true);
    },
    options: { preventDefault: true, enabled: isGalleryFocused },
    dependencies: [isOnLastRow, goNext, isNextEnabled, queryResult.isFetching, handleDownImage, isGalleryFocused],
  });

  useRegisteredHotkeys({
    id: 'deleteSelection',
    category: 'gallery',
    callback: () => {
      if (!selection.length) {
        return;
      }
      dispatch(imagesToDeleteSelected(selection));
    },
    options: {
      enabled: (isGalleryFocused || isImageViewerFocused) && isDeleteEnabledByTab && !isWorkflowsFocused,
    },
    dependencies: [isWorkflowsFocused, isDeleteEnabledByTab, selection, isWorkflowsFocused],
  });
};
