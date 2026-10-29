import type { GalleryBoard } from '@features/gallery/core/types';

import { galleryBoardsOptions } from '@features/gallery/data/queries';
import { canAddToCachedGalleryBoard, getCachedGalleryBoard } from '@features/gallery/data/queryCache';
import { useQuery, useQueryClient } from '@tanstack/react-query';

import { getGalleryIntendedDestinationId, type GalleryStateView } from './galleryStateView';

export interface GalleryDestination {
  /** The board new results go to now, as a submission resolves it. */
  boardId: string;
  /** The board they were meant for when that board refuses them (view-only, gone); null otherwise. */
  refusedBoardId: string | null;
  /** A listed board, else one any cached listing holds (an archived one, say); undefined for a board none holds. */
  getBoard: (boardId: string) => GalleryBoard | undefined;
}

/**
 * Where new results go, decided as submissions decide it (`getGalleryAutoAddBoardId` against the cached board
 * listings), for the surfaces that name it: the header's Saving to and the panel's Auto badge. A chosen board the
 * panel's listing lacks may be archived, so only then is the listing of archived boards read too; it is the one
 * submissions ensure, so the read is shared. Until it lands, or while the panel's own listing loads, the chosen board
 * is shown as chosen rather than refused.
 */
export const useGalleryDestination = (
  view: Pick<GalleryStateView, 'boards' | 'projectBoardId' | 'selectedBoardId' | 'settings'>
): GalleryDestination => {
  const queryClient = useQueryClient();
  const intendedId = getGalleryIntendedDestinationId(view);
  // Uncategorized and the project's inbox are always the caller's own; an empty list is one still loading.
  const isJudged = intendedId !== 'none' && intendedId !== view.projectBoardId && view.boards.length > 0;
  const needsArchivedListing = isJudged && !view.boards.some((board) => board.id === intendedId);
  const archivedListing = useQuery({
    ...galleryBoardsOptions({ includeArchived: true }),
    enabled: needsArchivedListing,
  });
  const isResolving = needsArchivedListing && archivedListing.isPending;
  const isRefused = isJudged && !isResolving && !canAddToCachedGalleryBoard(intendedId, queryClient);

  return {
    boardId: isRefused ? (view.projectBoardId ?? 'none') : intendedId,
    getBoard: (boardId) =>
      view.boards.find((board) => board.id === boardId) ?? getCachedGalleryBoard(boardId, queryClient),
    refusedBoardId: isRefused ? intendedId : null,
  };
};
