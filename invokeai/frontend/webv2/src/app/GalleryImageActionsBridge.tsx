import type { GalleryItemActionsOptions, GalleryItemContextMenuProps } from '@features/gallery/react';
import type { ReactNode } from 'react';

import { GalleryItemActionsProvider } from '@features/gallery/react';
import {
  ImageContextMenu,
  SelectedImageExportCommand,
  useDeletionConfirmation,
  useImageActions,
  useImageExportDialog,
  type ImageActions,
} from '@workbench/image-actions';
import { createContext, use } from 'react';
import { useTranslation } from 'react-i18next';

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
  runtime,
}: GalleryItemActionsOptions & { children: ReactNode }) => {
  const { i18n } = useTranslation();
  const { dialog, requestDeletionConfirmation } = useDeletionConfirmation();
  const { dialog: exportDialog, requestImageExport } = useImageExportDialog();
  const actions = useImageActions({
    boards,
    generateValues,
    getItemActionContext,
    projectId,
    requestDeletionConfirmation,
    requestImageExport,
  });

  return (
    <>
      <GalleryAppActionsContext value={actions}>
        <GalleryItemActionsProvider actions={actions}>{children}</GalleryItemActionsProvider>
      </GalleryAppActionsContext>
      {runtime ? (
        <SelectedImageExportCommand
          key={`${projectId}:${i18n.language}`}
          openImageExport={actions.openImageExport}
          runtime={runtime}
        />
      ) : null}
      {dialog}
      {exportDialog}
    </>
  );
};

export const GalleryItemActionsAdapter = GalleryItemActionsAdapterComponent;
export const GalleryImageContextMenu = GalleryImageContextMenuComponent;
