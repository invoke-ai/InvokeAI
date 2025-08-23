import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectImageToCompare, selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { VideoPlayer } from 'features/video/components/VideoPlayer';
import { memo } from 'react';

import { ImageViewerContextProvider } from './context';
import { ImageComparison } from './ImageComparison';
import { ImageViewer } from './ImageViewer';

const selectIsComparing = createSelector(
  [selectLastSelectedItem, selectImageToCompare],
  (lastSelectedImage, imageToCompare) => !!lastSelectedImage && !!imageToCompare
);

export const ImageViewerPanel = memo(() => {
  const isComparing = useAppSelector(selectIsComparing);
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);

  return (
    <ImageViewerContextProvider>
      {!isComparing && lastSelectedItem?.type === 'image' && <ImageViewer />}
      {!isComparing && lastSelectedItem?.type === 'video' && <VideoPlayer />}
      {isComparing && <ImageComparison />}
    </ImageViewerContextProvider>
  );
});
ImageViewerPanel.displayName = 'ImageViewerPanel';
