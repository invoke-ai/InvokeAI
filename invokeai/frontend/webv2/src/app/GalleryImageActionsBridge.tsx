import type { GalleryItemActionsOptions, GalleryItemContextMenuProps } from '@features/gallery/react';
import type { ReactNode } from 'react';

import { GalleryItemActionsProvider } from '@features/gallery/react';
import {
  ImageContextMenu,
  useDeletionConfirmation,
  useImageActions,
  type ImageActions,
} from '@workbench/image-actions';
import { createContext, use } from 'react';

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
