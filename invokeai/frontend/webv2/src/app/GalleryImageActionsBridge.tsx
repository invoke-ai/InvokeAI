import type {
  GalleryImageActions,
  GalleryImageActionsOptions,
  GalleryItemContextMenuProps,
} from '@features/gallery/react';
import type { ReactNode } from 'react';

import { galleryImageItemToGalleryImage, isGalleryImageItem, toGalleryItemKey } from '@features/gallery/contracts';
import { GalleryImageActionsProvider } from '@features/gallery/react';
import { getGalleryItemDragData, isGalleryImageDragData } from '@features/gallery/utility';
import { ImageContextMenu, useImageActions, type ImageActions } from '@workbench/image-actions';
import { useMemo } from 'react';

const GalleryImageContextMenuComponent = ({ actions, boards, onClose, target }: GalleryItemContextMenuProps) => {
  // TODO(Task 7): The current App action menu is image-only. Keep its input
  // strict until the canonical mixed-media context menu/actions port lands.
  const imageTarget = useMemo(() => {
    if (!target || target.itemRefs.length === 0 || target.itemRefs.some((ref) => ref.kind !== 'image')) {
      return null;
    }

    const loadedItemsByKey = new Map(target.items.map((item) => [toGalleryItemKey(item), item]));
    const imageItems = target.itemRefs.flatMap((ref) => {
      const item = loadedItemsByKey.get(toGalleryItemKey(ref));

      return item && isGalleryImageItem(item) ? [item] : [];
    });

    if (imageItems.length !== target.itemRefs.length) {
      return null;
    }

    return {
      images: imageItems.map(galleryImageItemToGalleryImage),
      x: target.x,
      y: target.y,
    };
  }, [target]);

  return (
    <ImageContextMenu
      actions={actions as unknown as ImageActions}
      boards={boards}
      target={imageTarget}
      onClose={onClose}
    />
  );
};

const GalleryImageActionsAdapterComponent = ({
  boards,
  children,
  generateValues,
  onImagesDeleted,
  projectId,
}: GalleryImageActionsOptions & { children: ReactNode }) => {
  const actions = useImageActions({ boards, generateValues, onImagesDeleted, projectId });
  const galleryActions = useMemo<GalleryImageActions>(
    () => ({
      ...actions,
      // TODO(Task 7): Replace with the canonical mixed-media organization
      // action. This compatibility bridge deliberately rejects video/mixed
      // refs instead of silently moving only their image subset.
      moveItemsToBoard: async (items, boardId) => {
        const dragData = getGalleryItemDragData(items);

        if (!isGalleryImageDragData(dragData)) {
          return;
        }

        await actions.moveImagesToBoard(
          dragData.items.map((item) => item.name),
          boardId
        );
      },
    }),
    [actions]
  );

  return <GalleryImageActionsProvider actions={galleryActions}>{children}</GalleryImageActionsProvider>;
};

export const GalleryImageActionsAdapter = GalleryImageActionsAdapterComponent;
export const GalleryImageContextMenu = GalleryImageContextMenuComponent;
