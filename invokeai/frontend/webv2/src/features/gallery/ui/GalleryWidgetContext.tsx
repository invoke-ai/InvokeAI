import type { GalleryImageItem, GalleryItem, GalleryItemRef } from '@features/gallery/contracts';
import type { GallerySettings } from '@features/gallery/core/settings';
import type { GalleryView } from '@features/gallery/core/types';

import { createContext, use } from 'react';

import type { GalleryStateView } from './galleryStateView';
import type { GalleryImageActions, GalleryWidgetRuntime } from './GalleryUiContext';

/**
 * Gallery-widget intents. The provider (GalleryWidgetView) is the only place
 * that knows whether an intent maps to a workbench dispatch, a backend call,
 * or both. Image-level operations (star, delete, move, ...) live in the shared
 * ImageActions contract instead so other widgets reuse them.
 */
export interface GalleryActions {
  archiveBoard: (boardId: string, archived: boolean) => Promise<void>;
  createBoard: (boardName: string) => Promise<void>;
  deleteBoard: (boardId: string, includeImages: boolean) => Promise<void>;
  downloadBoard: (boardId: string) => Promise<void>;
  loadMore: () => void;
  refresh: () => void;
  renameBoard: (boardId: string, boardName: string) => Promise<void>;
  selectBoard: (boardId: string) => void;
  selectItem: (item: GalleryItem) => void;
  selectItemRange: (items: GalleryItemRef[], primaryItem: GalleryItem) => void;
  selectProjectBoard: () => Promise<void>;
  setCompareItem: (image: GalleryImageItem | null) => void;
  setSearchTerm: (searchTerm: string) => void;
  setView: (galleryView: GalleryView) => void;
  toggleItemInSelection: (item: GalleryItem, nextPrimaryItem: GalleryItem | null) => void;
  updateSettings: (settings: Partial<GallerySettings>) => void;
  uploadFiles: (files: File[]) => Promise<void>;
}

export interface GalleryWidgetContextValue {
  gallery: GalleryStateView;
  actions: GalleryActions;
  imageActions: GalleryImageActions;
  /** The infinite window is full and the board holds images it cannot reach. */
  isWindowTruncated: boolean;
  projectName: string;
  runtime: GalleryWidgetRuntime;
}

export const GalleryWidgetContext = createContext<GalleryWidgetContextValue | null>(null);

export const useGalleryWidget = (): GalleryWidgetContextValue => {
  const value = use(GalleryWidgetContext);

  if (!value) {
    throw new Error('useGalleryWidget must be used within the gallery widget.');
  }

  return value;
};
