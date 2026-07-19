import type { GalleryImageActionsOptions, GalleryImageContextMenuProps } from '@features/gallery/react';
import type { ReactNode } from 'react';

import { GalleryImageActionsProvider } from '@features/gallery/react';
import { ImageContextMenu, useImageActions, type ImageActions } from '@workbench/image-actions';

export const GalleryImageContextMenu = ({ actions, ...props }: GalleryImageContextMenuProps) => (
  <ImageContextMenu {...props} actions={actions as ImageActions} />
);

export const GalleryImageActionsAdapter = ({
  boards,
  children,
  generateValues,
  onImagesDeleted,
  onStarredChange,
  projectId,
}: GalleryImageActionsOptions & { children: ReactNode }) => {
  const actions = useImageActions({ boards, generateValues, onImagesDeleted, onStarredChange, projectId });

  return <GalleryImageActionsProvider actions={actions}>{children}</GalleryImageActionsProvider>;
};
