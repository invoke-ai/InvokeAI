import type { GalleryItem, GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryItemActionsOptions, GalleryItemContextMenuProps } from '@features/gallery/react';
import type { ReactNode } from 'react';

import { galleryItems } from '@features/gallery';
import { galleryBoardsOptions } from '@features/gallery/queries';
import { GalleryItemActionsProvider, useHostGalleryBoards } from '@features/gallery/react';
import { useQuery } from '@tanstack/react-query';
import {
  ImageContextMenu,
  useDeletionConfirmation,
  useImageActions,
  type ImageActions,
} from '@workbench/image-actions';
import { createContext, use, useCallback } from 'react';

import type { GalleryItemMenuHostProps, LoadedGalleryItemMenu } from './galleryItemMenu';

const GalleryAppActionsContext = createContext<ImageActions | null>(null);

const GalleryImageContextMenuComponent = ({ boards, onClose, target }: GalleryItemContextMenuProps) => {
  const actions = use(GalleryAppActionsContext);

  if (!actions) {
    throw new Error('Gallery item context menus require the App-owned action adapter.');
  }

  return <ImageContextMenu actions={actions} boards={boards} target={target} onClose={onClose} />;
};

const GalleryItemActionsAdapterComponent = ({
  boards,
  children,
  generateValues,
  getItemActionContext,
  projectId,
}: GalleryItemActionsOptions & { children: ReactNode }) => {
  const { dialog, requestDeletionConfirmation } = useDeletionConfirmation();
  const actions = useImageActions({
    boards,
    generateValues,
    getItemActionContext,
    projectId,
    requestDeletionConfirmation,
  });

  return (
    <>
      <GalleryAppActionsContext value={actions}>
        <GalleryItemActionsProvider actions={actions}>{children}</GalleryItemActionsProvider>
      </GalleryAppActionsContext>
      {dialog}
    </>
  );
};

export const GalleryItemActionsAdapter = GalleryItemActionsAdapterComponent;
export const GalleryImageContextMenu = GalleryImageContextMenuComponent;

/** Mounted from the first opening on, so a deletion confirmation the menu opened survives the menu closing. */
const GalleryItemMenuHost = ({ generateValues, projectId, view, onClose }: GalleryItemMenuHostProps) => {
  const boards = useHostGalleryBoards(useQuery({ ...galleryBoardsOptions(), enabled: view !== null }).data);
  const viewId = view?.id;
  const handleClose = useCallback(() => {
    if (viewId !== undefined) {
      onClose(viewId);
    }
  }, [onClose, viewId]);

  return (
    <GalleryItemActionsAdapterComponent boards={boards} generateValues={generateValues} projectId={projectId}>
      <GalleryImageContextMenuComponent
        key={viewId}
        boards={boards}
        target={view?.target ?? null}
        onClose={handleClose}
      />
    </GalleryItemActionsAdapterComponent>
  );
};

export const loadGalleryItemMenu = async (
  ref: GalleryItemRef,
  item: GalleryItem | undefined,
  signal: AbortSignal
): Promise<LoadedGalleryItemMenu> => ({
  Host: GalleryItemMenuHost,
  item: item ?? (await galleryItems.resolve(ref, signal)),
});
