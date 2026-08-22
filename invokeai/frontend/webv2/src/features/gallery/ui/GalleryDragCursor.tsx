import { useDndContext } from '@dnd-kit/core';
import { useEffect } from 'react';

import { isGalleryItemDragData } from './galleryDnd';

/**
 * Marks `<body data-gallery-drag>` while any gallery-item drag is in flight;
 * the theme's global rule turns the whole app's cursor into the closed hand
 * for the duration (a body-level cursor alone would lose to every element
 * that sets its own — buttons, textareas — flickering arrow/I-beam back as
 * the pointer crosses them mid-drag). Mount once inside the DndContext.
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
