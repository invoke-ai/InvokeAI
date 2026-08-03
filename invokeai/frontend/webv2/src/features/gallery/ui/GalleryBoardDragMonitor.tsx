import { useDndMonitor, type DragEndEvent, type DragStartEvent } from '@dnd-kit/core';
import { useCallback, useRef } from 'react';

import { forwardGalleryBoardDrop, isGalleryItemDragData } from './galleryDnd';
import { useGalleryWidget } from './GalleryWidgetContext';

/**
 * Owns board-drop forwarding and the temporary disclosure needed to make
 * board targets available when an item drag begins from a collapsed gallery.
 */
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
      forwardGalleryBoardDrop({
        activeData: event.active.data.current,
        loadedItems: gallery.items,
        moveItemsToBoard: (items, boardId) => void itemActions.moveItemsToBoard(items, boardId),
        overData: event.over?.data.current,
      });
      restoreDisclosure();
    },
    [gallery.items, itemActions, restoreDisclosure]
  );

  useDndMonitor({
    onDragCancel: restoreDisclosure,
    onDragEnd: handleDragEnd,
    onDragStart: handleDragStart,
  });

  return null;
};
