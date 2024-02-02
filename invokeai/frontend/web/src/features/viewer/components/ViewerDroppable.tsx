import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import IAIDroppable from 'common/components/IAIDroppable';
import type { ViewerImageDropData } from 'features/dnd/types';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const selectLastSelectedImageName = createSelector(
  selectLastSelectedImage,
  (lastSelectedImage) => lastSelectedImage?.image_name
);

export const ViewerDroppable = memo(() => {
  const { t } = useTranslation();
  const currentImageName = useAppSelector(selectLastSelectedImageName);
  const viewerMode = useAppSelector((s) => s.viewer.viewerMode);
  const viewerDropData = useMemo<ViewerImageDropData>(
    () => ({
      id: 'viewer-image',
      actionType: 'SET_VIEWER_IMAGE',
      context: {
        currentImageName,
        viewerMode,
      },
    }),
    [currentImageName, viewerMode]
  );

  return <IAIDroppable data={viewerDropData} dropLabel={t('viewer.dropLabel')} />;
});

ViewerDroppable.displayName = 'ViewerDroppable';
