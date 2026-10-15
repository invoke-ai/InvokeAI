import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectImageToCompare, selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { memo } from 'react';

import { ImageViewerContextProvider } from './context';
import { ImageComparison } from './ImageComparison';
import { ImageViewer } from './ImageViewer';
import { useToggleMetadataHotkey } from './useToggleMetadataHotkey';

const selectIsComparing = createSelector(
  [selectLastSelectedItem, selectImageToCompare],
  (lastSelectedImage, imageToCompare) => !!lastSelectedImage && !!imageToCompare
);

export const ImageViewerPanel = memo(() => {
  return (
    <ImageViewerContextProvider>
      <ImageViewerPanelContent />
    </ImageViewerContextProvider>
  );
});
ImageViewerPanel.displayName = 'ImageViewerPanel';

const ImageViewerPanelContent = memo(() => {
  const isComparing = useAppSelector(selectIsComparing);
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  useToggleMetadataHotkey();

  return (
    <>
      {
        // The image viewer renders progress images - if no image is selected, show the image viewer anyway
        !isComparing && !lastSelectedItem && <ImageViewer />
      }
      {!isComparing && <ImageViewer />}
      {isComparing && <ImageComparison />}
    </>
  );
});
ImageViewerPanelContent.displayName = 'ImageViewerPanelContent';
