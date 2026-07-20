import type { GalleryImageActionsOptions, GalleryImageContextMenuProps } from '@features/gallery/react';
import type { ReactNode } from 'react';

import { GalleryImageActionsProvider } from '@features/gallery/react';
import { ImageContextMenu, useImageActions, type ImageActions } from '@workbench/image-actions';

import { bindProductionPortBinding } from './productionPortBindings';

const GalleryImageContextMenuComponent = ({ actions, ...props }: GalleryImageContextMenuProps) => (
  <ImageContextMenu {...props} actions={actions as ImageActions} />
);

const GalleryImageActionsAdapterComponent = ({
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

const galleryImageActionsBridge = bindProductionPortBinding('gallery.image-actions-bridge', {
  ImageActionsAdapter: GalleryImageActionsAdapterComponent,
  ImageContextMenu: GalleryImageContextMenuComponent,
});

export const GalleryImageActionsAdapter = galleryImageActionsBridge.ImageActionsAdapter;
export const GalleryImageContextMenu = galleryImageActionsBridge.ImageContextMenu;
