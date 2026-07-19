import type { GallerySettings } from '@features/gallery/core/settings';
import type { GalleryBoard, GalleryImage, GalleryView } from '@features/gallery/core/types';
import type { QueueItem } from '@features/queue/contracts';

import { createContext, use, type ComponentType, type ReactNode } from 'react';

export interface GalleryImageActions {
  deleteImages(imageNames: string[]): Promise<void>;
  moveImagesToBoard(imageNames: string[], boardId: string): Promise<void>;
  setImagesStarred(imageNames: string[], starred: boolean): Promise<void>;
}

export interface GalleryImageActionsOptions {
  boards: GalleryBoard[];
  generateValues: Record<string, unknown>;
  onImagesDeleted(imageNames: string[]): void;
  onStarredChange(imageNames: string[], starred: boolean): void;
  projectId: string;
}

export interface GalleryImageContextMenuTarget {
  images: GalleryImage[];
  x: number;
  y: number;
}

export interface GalleryImageContextMenuProps {
  actions: GalleryImageActions;
  boards: GalleryBoard[];
  target: GalleryImageContextMenuTarget | null;
  onClose(): void;
}

export interface GalleryCommandsPort {
  selectBoard(boardId: string): void;
  selectImage(image: GalleryImage): void;
  setCompareImage(image: GalleryImage | null): void;
  setMultiSelection(imageNames: string[], primaryImage: GalleryImage): void;
  setPage(page: number): void;
  setPageInfo(totalImages: number): void;
  setProjectBoard(boardId: string): void;
  setSearchTerm(searchTerm: string): void;
  setView(view: GalleryView): void;
  toggleImageSelection(image: GalleryImage): void;
  touch(): void;
  updateSettings(settings: Partial<GallerySettings>): void;
}

export interface GalleryNotificationsPort {
  add(notification: { kind: 'info' | 'success'; message?: string; title: string }): void;
  reportError(error: { area: string; message: string; namespace: 'gallery' }): void;
}

export interface GalleryWidgetRuntime {
  commands: {
    register(command: { handler: () => unknown; id: string; title: string }): () => void;
  };
  hotkeys: {
    register(hotkey: { commandId: string; defaultKeys: string[]; id: string; title: string }): () => void;
  };
}

export interface GalleryWidgetProps {
  presentation?: 'compact' | 'expanded' | 'tooltip';
  region: 'bottom' | 'center' | 'dialog' | 'left' | 'popover' | 'right';
  runtime: GalleryWidgetRuntime;
}

/**
 * Gallery's UI port. The context is a dependency-direction port (the feature
 * may not import workbench), not a test seam; no second adapter is expected.
 */
export interface GalleryUiAdapter {
  ImageActionsProvider: ComponentType<GalleryImageActionsOptions & { children: ReactNode }>;
  ImageContextMenu: ComponentType<GalleryImageContextMenuProps>;
  account: { showProgressImages(): void };
  antialiasProgressImages: boolean;
  gallery: GalleryCommandsPort;
  galleryValues: Record<string, unknown>;
  generateValues: Record<string, unknown>;
  notifications: GalleryNotificationsPort;
  projectId: string;
  projectName: string;
  queueItems: QueueItem[];
  widgets: { patchGalleryValues(values: Record<string, unknown>): void };
}

const GalleryUiContext = createContext<GalleryUiAdapter | null>(null);
const GalleryImageActionsContext = createContext<GalleryImageActions | null>(null);

export const GalleryImageActionsProvider = ({
  actions,
  children,
}: {
  actions: GalleryImageActions;
  children: ReactNode;
}) => <GalleryImageActionsContext value={actions}>{children}</GalleryImageActionsContext>;

export const useGalleryImageActions = (): GalleryImageActions => {
  const actions = use(GalleryImageActionsContext);

  if (!actions) {
    throw new Error('Gallery image actions require the App-owned image-actions adapter.');
  }

  return actions;
};

export const GalleryUiProvider = ({ adapter, children }: { adapter: GalleryUiAdapter; children: ReactNode }) => (
  <GalleryUiContext value={adapter}>{children}</GalleryUiContext>
);

export const useGalleryUi = (): GalleryUiAdapter => {
  const adapter = use(GalleryUiContext);

  if (!adapter) {
    throw new Error('Gallery UI requires an App-composed GalleryUiProvider.');
  }

  return adapter;
};
