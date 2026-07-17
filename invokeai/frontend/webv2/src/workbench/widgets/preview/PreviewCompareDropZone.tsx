import { Badge, Flex } from '@chakra-ui/react';
import { useDndContext, useDroppable } from '@dnd-kit/core';
import { isGalleryImageDragData } from '@workbench/widgets/gallery/galleryDnd';
import { useTranslation } from 'react-i18next';

import { PREVIEW_COMPARE_DROP_DATA, PREVIEW_COMPARE_DROP_ID } from './previewCompareDnd';

/**
 * A quiet drop ring over the preview frame, visible only while a gallery-image
 * drag is in flight. Dropping arms the dragged image for comparison (resolved
 * by the widget shell's dnd monitor).
 */
export const PreviewCompareDropZone = () => {
  const { t } = useTranslation();
  const { active } = useDndContext();
  const { isOver, setNodeRef } = useDroppable({ data: PREVIEW_COMPARE_DROP_DATA, id: PREVIEW_COMPARE_DROP_ID });

  if (!isGalleryImageDragData(active?.data.current)) {
    return null;
  }

  return (
    <Flex
      ref={setNodeRef}
      align="center"
      bg={isOver ? 'bg.subtle' : 'transparent'}
      borderColor={isOver ? 'border.emphasized' : 'border.subtle'}
      borderStyle="dashed"
      borderWidth="1px"
      inset="1"
      justify="center"
      opacity={isOver ? 1 : 0.85}
      position="absolute"
      rounded="lg"
      transitionDuration="var(--wb-motion-duration-fast)"
      transitionProperty="opacity, background-color, border-color"
      transitionTimingFunction="ease"
      zIndex="2"
    >
      <Badge size="xs" variant="subtle">
        {t('widgets.preview.dropToCompare')}
      </Badge>
    </Flex>
  );
};
