import { Badge } from '@chakra-ui/react';
import { useDndContext, useDroppable } from '@dnd-kit/core';
import { isGalleryImageDragData } from '@features/gallery/utility';
import { DropZone } from '@platform/ui/DropZone';
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
    <DropZone
      ref={setNodeRef}
      alignItems="center"
      display="flex"
      inset="1"
      isOver={isOver}
      justifyContent="center"
      opacity={isOver ? 1 : 0.85}
      position="absolute"
      rounded="lg"
      variant="overlay"
      zIndex="2"
    >
      <Badge size="xs" variant="subtle">
        {t('widgets.preview.dropToCompare')}
      </Badge>
    </DropZone>
  );
};
