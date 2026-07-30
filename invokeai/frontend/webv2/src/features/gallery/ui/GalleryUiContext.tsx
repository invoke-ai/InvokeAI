import type { GalleryImageItem, GalleryItem, GalleryItemKey, GalleryItemRef } from '@features/gallery/contracts';
import type { GallerySettings } from '@features/gallery/core/settings';
import type { GalleryBoard, GalleryBoardDeletionResult, GalleryImage, GalleryView } from '@features/gallery/core/types';
import type { QueueItem } from '@features/queue/contracts';

import { createContext, use, type ComponentType, type ReactNode } from 'react';

import type { GalleryLiveTarget } from './galleryStateView';

export interface GalleryImageActions {
  deleteImages(imageNames: string[]): Promise<void>;
  /**
   * TODO(Task 7): Move this narrow board-drop intent to `GalleryItemActions`
   * when common item mutations gain their mixed-media implementation.
   */
  moveItemsToBoard(items: GalleryItemRef[], boardId: string): Promise<void>;
  moveImagesToBoard(imageNames: string[], boardId: string): Promise<void>;
  setImagesStarred(imageNames: string[], starred: boolean): Promise<void>;
}

export interface GalleryImageActionsOptions {
  boards: GalleryBoard[];
  generateValues: Record<string, unknown>;
  onImagesDeleted(imageNames: string[]): void;
  projectId: string;
}

export interface GalleryItemContextMenuTarget {
  itemRefs: GalleryItemRef[];
  items: GalleryItem[];
  x: number;
  y: number;
}

export interface GalleryItemContextMenuProps {
  actions: GalleryImageActions;
  boards: GalleryBoard[];
  target: GalleryItemContextMenuTarget | null;
  onClose(): void;
}

export interface GalleryCommandsPort {
  reconcileDeletedBoardOutcome(outcome: GalleryBoardDeletionResult): void;
  selectBoard(boardId: string): void;
  selectItem(item: GalleryItem): void;
  selectImage(image: GalleryImage): void;
  setCompareItem(image: GalleryImageItem | null): void;
  setCompareImage(image: GalleryImage | null): void;
  setItemMultiSelection(itemKeys: GalleryItemKey[], primaryItem: GalleryItem): void;
  setMultiSelection(imageNames: string[], primaryImage: GalleryImage): void;
  setPage(page: number): void;
  setPageInfo(totalImages: number): void;
  setProjectBoard(boardId: string): void;
  setSearchTerm(searchTerm: string): void;
  setView(view: GalleryView): void;
  toggleItemSelection(item: GalleryItem, nextPrimaryItem: GalleryItem | null): void;
  toggleImageSelection(image: GalleryImage, nextPrimaryItem: GalleryItem | null): void;
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
  ImageContextMenu: ComponentType<GalleryItemContextMenuProps>;
  account: { enableLiveFollow(): void };
  antialiasProgressImages: boolean;
  gallery: GalleryCommandsPort;
  galleryValues: Record<string, unknown>;
  generateValues: Record<string, unknown>;
  notifications: GalleryNotificationsPort;
  projectId: string;
  projectName: string;
  queueItems: QueueItem[];
  liveFollowEnabled: boolean;
  liveProgressTarget: GalleryLiveTarget | null;
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
