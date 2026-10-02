import type { GalleryImageItem, GalleryItem, GalleryItemRef } from '@features/gallery/contracts';
import type { GallerySemanticReference } from '@features/gallery/core/semanticImageQuery';
import type { GallerySettings } from '@features/gallery/core/settings';
import type { GalleryView } from '@features/gallery/core/types';
import type { GalleryItemsFilter } from '@features/gallery/data/queries';

import { createContext, use } from 'react';

import type { GalleryStateView } from './galleryStateView';
import type { GalleryItemActions, GalleryWidgetProps, GalleryWidgetRuntime } from './GalleryUiContext';

/**
 * The provider maps widget intents to workbench/backend actions; shared ImageActions owns cross-widget item
 * operations.
 */
export interface GalleryActions {
  archiveBoard: (boardId: string, archived: boolean) => Promise<void>;
  createBoard: (boardName: string) => Promise<void>;
  deleteBoard: (boardId: string, includeImages: boolean) => Promise<void>;
  downloadBoard: (boardId: string) => Promise<void>;
  /** Export the project that owns this board as a complete `.invk` archive. */
  exportProject: (projectId: string, projectName: string) => void;
  loadMore: () => void;
  refresh: () => void;
  renameBoard: (boardId: string, boardName: string) => Promise<void>;
  selectBoard: (boardId: string) => void;
  selectItem: (item: GalleryItem) => void;
  selectItemRange: (items: GalleryItemRef[], primaryItem: GalleryItem) => void;
  setCompareItem: (image: GalleryImageItem | null) => void;
  setSearchTerm: (searchTerm: string) => void;
  /** Restricts (or releases) the listing to starred items; resets the page like a search. */
  setStarredOnly: (starredOnly: boolean) => void;
  /** Clears the search field: its text, any ranking, and semantic mode. */
  clearSearch: () => void;
  /** Applies the semantic field's text as the ranking; a no-op once the field has moved on. */
  commitSemanticSearch: (text: string) => void;
  /** Sets (or clears) the image-similarity query shown as a chip in the search field. */
  setSemanticImageQuery: (reference: GallerySemanticReference | null) => void;
  /** Switches the search field between metadata and semantic search, keeping its text. */
  setSemanticSearchMode: (enabled: boolean) => void;
  /** The semantic field's live text, ahead of the debounced commit. */
  setSemanticSearchText: (text: string) => void;
  setView: (galleryView: GalleryView) => void;
  toggleItemInSelection: (item: GalleryItem, nextPrimaryItem: GalleryItem | null) => void;
  updateSettings: (settings: Partial<GallerySettings>) => void;
  /** Resolves with the confirmed uploads; empty when nothing landed. */
  uploadFiles: (files: File[]) => Promise<GalleryItem[]>;
}

/** The bounded starred strip above the listing; empty whenever it does not apply. */
export interface GalleryStarredStrip {
  items: GalleryItem[];
  /** Starred items under the same filter, per the backend; 0 until known. */
  total: number;
}

export interface GalleryWidgetContextValue {
  gallery: GalleryStateView;
  actions: GalleryActions;
  /**
   * The query filter the visible items came from. Shared rather than re-derived
   * so range selection and the item list can never disagree about which query
   * they are operating on.
   */
  filter: GalleryItemsFilter;
  itemActions: GalleryItemActions;
  /** The infinite window is full and the board holds images it cannot reach. */
  isWindowTruncated: boolean;
  /** Everything on hand — strip first, then the listing, without repeats — for lookups by key. */
  loadedItems: GalleryItem[];
  starredStrip: GalleryStarredStrip;
  projectName: string;
  /** Placement, used only to scope cached viewport measurements. */
  region: GalleryWidgetProps['region'];
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
