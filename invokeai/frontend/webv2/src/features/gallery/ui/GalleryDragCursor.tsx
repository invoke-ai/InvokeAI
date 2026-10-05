import { useDndContext } from '@dnd-kit/core';
import { useEffect } from 'react';

import { isGalleryItemDragData } from './galleryDnd';

/**
 * Mount once inside DndContext. The body marker enables global drag cursors that override element-specific button
 * and textarea cursors.
 */
export const GalleryDragCursor = () => {
  const { active } = useDndContext();
  const isGalleryDrag = isGalleryItemDragData(active?.data.current);

  useEffect(() => {
    if (!isGalleryDrag) {
      return;
    }

    document.body.setAttribute('data-gallery-drag', '');

    return () => document.body.removeAttribute('data-gallery-drag');
  }, [isGalleryDrag]);

  return null;
};
