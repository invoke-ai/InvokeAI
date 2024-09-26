import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { $focusedRegion, FOCUS_REGIONS } from 'common/hooks/interactionScopes';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { $canvasRightPanelTab } from 'features/controlLayers/store/ephemeral';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import { useGalleryNavigation } from 'features/gallery/hooks/useGalleryNavigation';
import { useGalleryPagination } from 'features/gallery/hooks/useGalleryPagination';
import { selectListImagesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
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
  const canvasRightPanelTab = useStore($canvasRightPanelTab);
  const appTab = useAppSelector(selectActiveTab);
  const workflowsFocus = useStore(FOCUS_REGIONS.$workflows);
  const focusedRegion = useStore($focusedRegion);

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
    options: { preventDefault: true, enabled: focusedRegion === 'galleryPanel' || focusedRegion === 'imageViewer' },
    dependencies: [handleLeftImage, isOnFirstImageOfView, goPrev, isPrevEnabled, queryResult.isFetching, focusedRegion],
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
    options: { preventDefault: true, enabled: focusedRegion === 'galleryPanel' || focusedRegion === 'imageViewer' },
    dependencies: [isOnLastImageOfView, goNext, isNextEnabled, queryResult.isFetching, handleRightImage, focusedRegion],
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
    options: { preventDefault: true, enabled: focusedRegion === 'galleryPanel' },
    dependencies: [handleUpImage, isOnFirstRow, goPrev, isPrevEnabled, queryResult.isFetching, focusedRegion],
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
    options: { preventDefault: true, enabled: focusedRegion === 'galleryPanel' },
    dependencies: [isOnLastRow, goNext, isNextEnabled, queryResult.isFetching, handleDownImage, focusedRegion],
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
    options: { preventDefault: true, enabled: focusedRegion === 'galleryPanel' || focusedRegion === 'imageViewer' },
    dependencies: [handleLeftImage, isOnFirstImageOfView, goPrev, isPrevEnabled, queryResult.isFetching, focusedRegion],
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
    options: { preventDefault: true, enabled: focusedRegion === 'galleryPanel' || focusedRegion === 'imageViewer' },
    dependencies: [isOnLastImageOfView, goNext, isNextEnabled, queryResult.isFetching, handleRightImage, focusedRegion],
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
    options: { preventDefault: true, enabled: focusedRegion === 'galleryPanel' },
    dependencies: [handleUpImage, isOnFirstRow, goPrev, isPrevEnabled, queryResult.isFetching, focusedRegion],
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
    options: { preventDefault: true, enabled: focusedRegion === 'galleryPanel' },
    dependencies: [isOnLastRow, goNext, isNextEnabled, queryResult.isFetching, handleDownImage, focusedRegion],
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
      enabled:
        (focusedRegion === 'galleryPanel' || focusedRegion === 'imageViewer') &&
        isDeleteEnabledByTab &&
        !workflowsFocus.isFocused,
    },
    dependencies: [focusedRegion, isDeleteEnabledByTab, selection, workflowsFocus],
  });
};
