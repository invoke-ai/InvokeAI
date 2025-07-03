import { Divider, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import type { SetComparisonImageDndTargetData } from 'features/dnd/dnd';
import { setComparisonImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { selectImageToCompare, selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { ImageViewerContextProvider } from './context';
import { ImageViewer } from './ImageViewer';
import { ViewerToolbar } from './ViewerToolbar';

export const ImageViewerPanel = memo(() => {
  const { t } = useTranslation();
  const lastSelectedImage = useAppSelector(selectLastSelectedImage);
  const imageToCompare = useAppSelector(selectImageToCompare);

  // Only show drop target when we have a selected image but no comparison image yet
  const shouldShowDropTarget = lastSelectedImage && !imageToCompare;

  const dndTargetData = useMemo<SetComparisonImageDndTargetData>(() => setComparisonImageDndTarget.getData(), []);

  return (
    <ImageViewerContextProvider>
      <Flex flexDir="column" w="full" h="full" overflow="hidden" gap={2} position="relative">
        <ViewerToolbar />
        <Divider />
        <Flex w="full" h="full" position="relative">
          <ImageViewer />
          {shouldShowDropTarget && (
            <DndDropTarget
              dndTarget={setComparisonImageDndTarget}
              dndTargetData={dndTargetData}
              label={t('gallery.selectForCompare')}
            />
          )}
        </Flex>
      </Flex>
    </ImageViewerContextProvider>
  );
});
ImageViewerPanel.displayName = 'ImageViewerPanel';
