import { Divider, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useGalleryItemDTO } from 'common/hooks/useGalleryItemDTO';
import { setComparisonImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { CurrentCanvasProjectPreview } from 'features/gallery/components/ImageViewer/CurrentCanvasProjectPreview';
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

  // Polymorphic preview: videos render the play-overlay/HTML5 video; canvas projects render the
  // preview WebP with a load-to-canvas affordance in the toolbar; images render the existing
  // DndImage-based preview with progress / metadata / next-prev affordances.
  let preview;
  if (galleryItem?.kind === 'video') {
    preview = <CurrentVideoPreview videoDTO={galleryItem.dto} />;
  } else if (galleryItem?.kind === 'canvas_project') {
    preview = <CurrentCanvasProjectPreview projectDTO={galleryItem.dto} />;
  } else {
    preview = <CurrentImagePreview imageDTO={galleryItem?.dto ?? null} />;
  }

  const isImage = galleryItem?.kind !== 'video' && galleryItem?.kind !== 'canvas_project';

  return (
    <Flex flexDir="column" w="full" h="full" overflow="hidden" gap={2} position="relative">
      <ImageViewerToolbar />
      <Divider />
      <Flex w="full" h="full" position="relative">
        {preview}
        {isImage && (
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
