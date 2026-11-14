import type { QueryClient } from '@tanstack/react-query';

import { getGalleryAutoAddBoardId } from '@features/gallery/ui/galleryStateView';

import { canAddToCachedGalleryBoard, ensureGalleryBoardDestinations } from './queryCache';

/**
 * Where media uploaded from outside the Gallery lands, such as a file dropped on Canvas or Preview: the project's
 * auto-add board, as generation results do, judged from the listing of every readable board, which is read first
 * when none is cached; Uncategorized when no board takes it. An auto-add board is never redirected for want of a
 * listing.
 */
export const resolveGalleryUploadBoardId = async (
  galleryValues: Record<string, unknown>,
  client: QueryClient
): Promise<string> => {
  await ensureGalleryBoardDestinations(client);
  return getGalleryAutoAddBoardId(galleryValues, (boardId) => canAddToCachedGalleryBoard(boardId, client)) ?? 'none';
};
