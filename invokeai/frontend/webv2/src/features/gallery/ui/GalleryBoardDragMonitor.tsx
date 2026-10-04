import { useDndMonitor, type DragEndEvent, type DragStartEvent } from '@dnd-kit/core';
import { useCallback, useRef } from 'react';

import { forwardGalleryBoardDrop, isGalleryItemDragData, resolveGallerySemanticSearchDrop } from './galleryDnd';
import { useGalleryWidget } from './GalleryWidgetContext';

/** Forward board drops and temporarily expose collapsed board targets during item drags. */
export const GalleryBoardDragMonitor = () => {
  const { actions, gallery, itemActions } = useGalleryWidget();
  const dragOpenedPanelRef = useRef(false);

  const restoreDisclosure = useCallback(() => {
    if (!dragOpenedPanelRef.current) {
      return;
    }

    dragOpenedPanelRef.current = false;
    actions.updateSettings({ boardPanelCollapsed: true });
  }, [actions]);

  const handleDragStart = useCallback(
    (event: DragStartEvent) => {
      if (!gallery.settings.boardPanelCollapsed || !isGalleryItemDragData(event.active.data.current)) {
        return;
      }

      dragOpenedPanelRef.current = true;
      actions.updateSettings({ boardPanelCollapsed: false });
    },
    [actions, gallery.settings.boardPanelCollapsed]
  );

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      const semanticDrop = resolveGallerySemanticSearchDrop(event.active.data.current, event.over?.id);

      if (semanticDrop) {
        actions.setSemanticImageQuery({ imageName: semanticDrop.imageName, kind: 'image' });
        restoreDisclosure();
        return;
      }

      forwardGalleryBoardDrop({
        activeData: event.active.data.current,
        loadedItems: gallery.items,
        moveItemsToBoard: (items, boardId) => void itemActions.moveItemsToBoard(items, boardId),
        overData: event.over?.data.current,
      });
      restoreDisclosure();
    },
    [actions, gallery.items, itemActions, restoreDisclosure]
  );

  useDndMonitor({
    onDragCancel: restoreDisclosure,
    onDragEnd: handleDragEnd,
    onDragStart: handleDragStart,
  });

  return null;
};
