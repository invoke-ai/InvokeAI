import { Divider, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { setComparisonImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { CurrentImagePreview } from 'features/gallery/components/ImageViewer/CurrentImagePreview';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useImageDTO } from 'services/api/endpoints/images';

import { ImageViewerToolbar } from './ImageViewerToolbar';

const dndTargetData = setComparisonImageDndTarget.getData();

export const ImageViewer = memo(() => {
  const { t } = useTranslation();

  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const lastSelectedImageDTO = useImageDTO(lastSelectedItem ?? null);
  return (
    <Flex flexDir="column" w="full" h="full" overflow="hidden" gap={2} position="relative">
      <ImageViewerToolbar />
      <Divider />
      <Flex w="full" h="full" position="relative">
        <CurrentImagePreview imageDTO={lastSelectedImageDTO} />
        <DndDropTarget
          dndTarget={setComparisonImageDndTarget}
          dndTargetData={dndTargetData}
          label={t('gallery.selectForCompare')}
        />
      </Flex>
    </Flex>
  );
});

ImageViewer.displayName = 'ImageViewer';
