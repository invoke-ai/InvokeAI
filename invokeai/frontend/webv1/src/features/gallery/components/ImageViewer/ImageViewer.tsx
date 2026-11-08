import { Divider, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useGalleryItemDTO } from 'common/hooks/useGalleryItemDTO';
import { setComparisonImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { CurrentImagePreview } from 'features/gallery/components/ImageViewer/CurrentImagePreview';
import { CurrentVideoPreview } from 'features/gallery/components/ImageViewer/CurrentVideoPreview';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import { ImageViewerToolbar } from './ImageViewerToolbar';

const dndTargetData = setComparisonImageDndTarget.getData();

export const ImageViewer = memo(() => {
  const { t } = useTranslation();

  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const galleryItem = useGalleryItemDTO(lastSelectedItem);

  // Polymorphic preview: videos render the play-overlay/HTML5 video; images render the existing
  // DndImage-based preview with progress / metadata / next-prev affordances.
  let preview;
  if (galleryItem?.kind === 'video') {
    preview = <CurrentVideoPreview videoDTO={galleryItem.dto} />;
  } else {
    preview = <CurrentImagePreview imageDTO={galleryItem?.dto ?? null} />;
  }

  return (
    <Flex flexDir="column" w="full" h="full" overflow="hidden" gap={2} position="relative">
      <ImageViewerToolbar />
      <Divider />
      <Flex w="full" h="full" position="relative">
        {preview}
        {galleryItem?.kind !== 'video' && (
          <DndDropTarget
            dndTarget={setComparisonImageDndTarget}
            dndTargetData={dndTargetData}
            label={t('gallery.selectForCompare')}
          />
        )}
      </Flex>
    </Flex>
  );
});

ImageViewer.displayName = 'ImageViewer';
