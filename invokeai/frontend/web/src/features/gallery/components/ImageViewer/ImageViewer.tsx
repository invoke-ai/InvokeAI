import { Divider, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { setComparisonImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import ImageMetadataViewer from 'features/gallery/components/ImageMetadataViewer/ImageMetadataViewer';
import { CurrentImagePreview } from 'features/gallery/components/ImageViewer/CurrentImagePreview';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { selectShouldShowItemDetails } from 'features/ui/store/uiSelectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useImageDTO } from 'services/api/endpoints/images';

import { ImageViewerToolbar } from './ImageViewerToolbar';

const dndTargetData = setComparisonImageDndTarget.getData();

export const ImageViewer = memo(() => {
  const { t } = useTranslation();
  const shouldShowItemDetails = useAppSelector(selectShouldShowItemDetails);

  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const lastSelectedImageDTO = useImageDTO(lastSelectedItem ?? null);
  return (
    <Flex flexDir="column" w="full" h="full" overflow="hidden" gap={2} position="relative">
      <ImageViewerToolbar />
      <Divider />
      <Flex w="full" h="full" position="relative" gap={2}>
        <Flex w="full" h="full" position="relative" flex={1} minW={0}>
          <CurrentImagePreview imageDTO={lastSelectedImageDTO} />
          <DndDropTarget
            dndTarget={setComparisonImageDndTarget}
            dndTargetData={dndTargetData}
            label={t('gallery.selectForCompare')}
          />
        </Flex>
        {shouldShowItemDetails && lastSelectedImageDTO && (
          <Flex w="400px" h="full" flexShrink={0} borderRadius="base" overflow="hidden">
            <ImageMetadataViewer image={lastSelectedImageDTO} />
          </Flex>
        )}
      </Flex>
    </Flex>
  );
});

ImageViewer.displayName = 'ImageViewer';
