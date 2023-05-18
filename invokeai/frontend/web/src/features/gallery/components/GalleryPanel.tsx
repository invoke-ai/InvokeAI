import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { gallerySelector } from 'features/gallery/store/gallerySelectors';
import { setGalleryImageMinimumWidth } from 'features/gallery/store/gallerySlice';

import { clamp, isEqual } from 'lodash-es';
import { useHotkeys } from 'react-hotkeys-hook';

import './ImageGallery.css';
import ImageGalleryContent from './ImageGalleryContent';
import ResizableDrawer from 'features/ui/components/common/ResizableDrawer/ResizableDrawer';
import { setShouldShowGallery } from 'features/ui/store/uiSlice';
import { createSelector } from '@reduxjs/toolkit';
import {
  activeTabNameSelector,
  uiSelector,
} from 'features/ui/store/uiSelectors';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import { lightboxSelector } from 'features/lightbox/store/lightboxSelectors';
import { memo } from 'react';

const selector = createSelector(
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

const GalleryDrawer = () => {
  const dispatch = useAppDispatch();
  const {
    shouldPinGallery,
    shouldShowGallery,
    galleryImageMinimumWidth,
    // activeTabName,
    // isStaging,
    // isResizable,
    // isLightboxOpen,
  } = useAppSelector(selector);

  const handleCloseGallery = () => {
    dispatch(setShouldShowGallery(false));
    shouldPinGallery && dispatch(requestCanvasRescale());
  };

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

  if (shouldPinGallery) {
    return null;
  }

  return (
    <ResizableDrawer
      direction="right"
      isResizable={true}
      isOpen={shouldShowGallery}
      onClose={handleCloseGallery}
      minWidth={200}
    >
      <ImageGalleryContent />
    </ResizableDrawer>
  );
};

export default memo(GalleryDrawer);
