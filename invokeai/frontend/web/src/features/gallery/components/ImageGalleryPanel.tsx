import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { gallerySelector } from 'features/gallery/store/gallerySelectors';
import {
  selectNextImage,
  selectPrevImage,
  setGalleryImageMinimumWidth,
} from 'features/gallery/store/gallerySlice';
import { InvokeTabName } from 'features/ui/store/tabMap';

import { clamp, isEqual } from 'lodash';
import { useHotkeys } from 'react-hotkeys-hook';

import './ImageGallery.css';
import ImageGalleryContent from './ImageGalleryContent';
import ResizableDrawer from 'features/ui/components/common/ResizableDrawer/ResizableDrawer';
import {
  setShouldShowGallery,
  toggleGalleryPanel,
  togglePinGalleryPanel,
} from 'features/ui/store/uiSlice';
import { createSelector } from '@reduxjs/toolkit';
import {
  activeTabNameSelector,
  uiSelector,
} from 'features/ui/store/uiSelectors';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import { lightboxSelector } from 'features/lightbox/store/lightboxSelectors';

const GALLERY_TAB_WIDTHS: Record<
  InvokeTabName,
  { galleryMinWidth: number; galleryMaxWidth: number }
> = {
  txt2img: { galleryMinWidth: 200, galleryMaxWidth: 500 },
  img2img: { galleryMinWidth: 200, galleryMaxWidth: 500 },
  unifiedCanvas: { galleryMinWidth: 200, galleryMaxWidth: 200 },
  nodes: { galleryMinWidth: 200, galleryMaxWidth: 500 },
  postprocessing: { galleryMinWidth: 200, galleryMaxWidth: 500 },
  training: { galleryMinWidth: 200, galleryMaxWidth: 500 },
};

const galleryPanelSelector = createSelector(
  [
    activeTabNameSelector,
    uiSelector,
    gallerySelector,
    isStagingSelector,
    lightboxSelector,
  ],
  (activeTabName, ui, gallery, isStaging, lightbox) => {
    const { shouldPinGallery, shouldShowGallery } = ui;
    const { galleryImageMinimumWidth } = gallery;
    const { isLightboxOpen } = lightbox;

    return {
      activeTabName,
      isStaging,
      shouldPinGallery,
      shouldShowGallery,
      galleryImageMinimumWidth,
      isResizable: activeTabName !== 'unifiedCanvas',
      isLightboxOpen,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export default function ImageGalleryPanel() {
  const dispatch = useAppDispatch();
  const {
    shouldPinGallery,
    shouldShowGallery,
    galleryImageMinimumWidth,
    activeTabName,
    isStaging,
    isResizable,
    isLightboxOpen,
  } = useAppSelector(galleryPanelSelector);

  const handleSetShouldPinGallery = () => {
    dispatch(togglePinGalleryPanel());
    dispatch(requestCanvasRescale());
  };

  const handleToggleGallery = () => {
    dispatch(toggleGalleryPanel());
    shouldPinGallery && dispatch(requestCanvasRescale());
  };

  const handleCloseGallery = () => {
    dispatch(setShouldShowGallery(false));
    shouldPinGallery && dispatch(requestCanvasRescale());
  };

  useHotkeys(
    'g',
    () => {
      handleToggleGallery();
    },
    [shouldPinGallery]
  );

  useHotkeys(
    'left',
    () => {
      dispatch(selectPrevImage());
    },
    {
      enabled: !isStaging || activeTabName !== 'unifiedCanvas',
    },
    [isStaging, activeTabName]
  );

  useHotkeys(
    'right',
    () => {
      dispatch(selectNextImage());
    },
    {
      enabled: !isStaging || activeTabName !== 'unifiedCanvas',
    },
    [isStaging, activeTabName]
  );

  useHotkeys(
    'shift+g',
    () => {
      handleSetShouldPinGallery();
    },
    [shouldPinGallery]
  );

  useHotkeys(
    'esc',
    () => {
      dispatch(setShouldShowGallery(false));
    },
    {
      enabled: () => !shouldPinGallery,
      preventDefault: true,
    },
    [shouldPinGallery]
  );

  const IMAGE_SIZE_STEP = 32;

  useHotkeys(
    'shift+up',
    () => {
      if (galleryImageMinimumWidth < 256) {
        const newMinWidth = clamp(
          galleryImageMinimumWidth + IMAGE_SIZE_STEP,
          32,
          256
        );
        dispatch(setGalleryImageMinimumWidth(newMinWidth));
      }
    },
    [galleryImageMinimumWidth]
  );

  useHotkeys(
    'shift+down',
    () => {
      if (galleryImageMinimumWidth > 32) {
        const newMinWidth = clamp(
          galleryImageMinimumWidth - IMAGE_SIZE_STEP,
          32,
          256
        );
        dispatch(setGalleryImageMinimumWidth(newMinWidth));
      }
    },
    [galleryImageMinimumWidth]
  );

  return (
    <ResizableDrawer
      direction="right"
      isResizable={isResizable || !shouldPinGallery}
      isOpen={shouldShowGallery}
      onClose={handleCloseGallery}
      isPinned={shouldPinGallery && !isLightboxOpen}
      minWidth={
        shouldPinGallery
          ? GALLERY_TAB_WIDTHS[activeTabName].galleryMinWidth
          : 200
      }
      maxWidth={
        shouldPinGallery
          ? GALLERY_TAB_WIDTHS[activeTabName].galleryMaxWidth
          : undefined
      }
    >
      <ImageGalleryContent />
    </ResizableDrawer>
  );
}
