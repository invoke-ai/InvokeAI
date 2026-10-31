import type { GallerySettings } from '@features/gallery/core/settings';
import type { GalleryView } from '@features/gallery/core/types';

import { getSelectedGalleryItemFromValues } from '@features/gallery/core/selection';
import { getGallerySettings } from '@features/gallery/core/settings';
import {
  getGalleryProjectBoardId,
  getGalleryRawSelectedBoardId,
  getGalleryView,
} from '@features/gallery/ui/galleryStateView';
import { useGalleryHost } from '@features/gallery/ui/GalleryUiContext';
import { useGalleryData } from '@features/gallery/ui/useGalleryData';
import { useCallback, useDeferredValue, useMemo, useState } from 'react';

export type GalleryPickerPane = 'boards' | 'items';

const NO_RECENT_IMAGES: never[] = [];

export interface GalleryPickerScope {
  boardId: string | null;
  galleryView: GalleryView;
  pane: GalleryPickerPane;
  searchTerm: string;
}

/**
 * Seed local scope on open without writing back. Clear search when switching panes because each pane interprets it
 * differently.
 */
export const useGalleryPickerScope = () => {
  const { galleryValues } = useGalleryHost();
  const [scope, setScope] = useState<GalleryPickerScope>(() => ({
    boardId: getGalleryRawSelectedBoardId(galleryValues),
    galleryView: getGalleryView(galleryValues),
    pane: 'items',
    searchTerm: '',
  }));
  const deferredSearchTerm = useDeferredValue(scope.searchTerm);
  const settings = useMemo<GallerySettings>(
    () => ({ ...getGallerySettings(galleryValues), paginationMode: 'infinite' }),
    [galleryValues]
  );
  const data = useGalleryData({
    galleryView: scope.galleryView,
    // Scope changes keep the previous list on screen, dimmed, until the new one lands or fails.
    keepPreviousScope: true,
    page: 0,
    projectBoardId: getGalleryProjectBoardId(galleryValues),
    // No recents overlay: `items` must stay null until the first page lands,
    // so the loading, seeding and stale states have one unambiguous signal.
    recentImages: NO_RECENT_IMAGES,
    searchTerm: scope.pane === 'items' ? deferredSearchTerm : '',
    selectedBoardId: scope.boardId,
    semanticQuery: null,
    sparseViewport: true,
    settings,
  });
  const gallerySelectedItem = useMemo(() => getSelectedGalleryItemFromValues(galleryValues), [galleryValues]);

  const setSearchTerm = useCallback((searchTerm: string) => setScope((current) => ({ ...current, searchTerm })), []);
  const setView = useCallback((galleryView: GalleryView) => setScope((current) => ({ ...current, galleryView })), []);
  const selectBoard = useCallback(
    (boardId: string) => setScope((current) => ({ ...current, boardId, pane: 'items', searchTerm: '' })),
    []
  );
  const togglePane = useCallback(
    () => setScope((current) => ({ ...current, pane: current.pane === 'items' ? 'boards' : 'items', searchTerm: '' })),
    []
  );

  return { data, gallerySelectedItem, scope, selectBoard, setSearchTerm, setView, settings, togglePane };
};
