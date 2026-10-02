import { Badge } from '@chakra-ui/react';
import { useDndContext, useDroppable } from '@dnd-kit/core';
import { isGalleryImageDragData } from '@features/gallery/utility';
import { DropZone } from '@platform/ui/DropZone';
import { useTranslation } from 'react-i18next';

import { PREVIEW_COMPARE_DROP_DATA, PREVIEW_COMPARE_DROP_ID } from './previewCompareDnd';

/**
 * Show comparison drop targets for image-only drags; reject the first item if already displayed so self-comparison
 * cannot pause live-follow.
 */
export const PreviewCompareDropZone = ({ currentImageName }: { currentImageName: string | null }) => {
  const { t } = useTranslation();
  const { active } = useDndContext();
  const activeData = active?.data.current;
  const isCompatible = isGalleryImageDragData(activeData);
  const isSelfCompare = isCompatible && activeData.items[0]?.name === currentImageName;
  const { isOver, setNodeRef } = useDroppable({
    data: PREVIEW_COMPARE_DROP_DATA,
    disabled: !isCompatible || isSelfCompare,
    id: PREVIEW_COMPARE_DROP_ID,
  });

  if (!isCompatible || isSelfCompare) {
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
