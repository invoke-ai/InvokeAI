import { useDndContext } from '@dnd-kit/core';
import { useEffect } from 'react';

import { isGalleryBoardDragData, isGalleryItemDragData } from './galleryDnd';

/** Mount once inside DndContext: the body marker's global drag cursors override button and textarea cursors. */
export const GalleryDragCursor = () => {
  const { active } = useDndContext();
  const isGalleryDrag = isGalleryItemDragData(active?.data.current) || isGalleryBoardDragData(active?.data.current);

  useEffect(() => {
    if (!isGalleryDrag) {
      return;
    }

    document.body.setAttribute('data-gallery-drag', '');

    return () => document.body.removeAttribute('data-gallery-drag');
  }, [isGalleryDrag]);

  return null;
};
