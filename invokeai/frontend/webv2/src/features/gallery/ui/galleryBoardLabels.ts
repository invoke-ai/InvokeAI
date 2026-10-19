import type { GalleryBoard, GalleryView } from '@features/gallery/core/types';

import { getBoardCounts } from './galleryStateView';

export const getGalleryCountForView = (board: GalleryBoard, galleryView: GalleryView): number => {
  const counts = getBoardCounts(board);

  // Uploaded videos count as Assets and generated videos as Media. Clamp parallel-request counts because
  // intervening deletions can temporarily invert totals.
  return galleryView === 'assets'
    ? counts.assetCount + counts.assetVideoCount
    : Math.max(0, counts.imageCount + counts.videoCount - counts.assetVideoCount);
};

type Translate = (key: string, options?: Record<string, unknown>) => string;

/**
 * Whose board someone else's board is and what the caller may do with it, for its row: "{owner} · Can view". Nothing
 * for the caller's own boards. A board the server reports no access for keeps its owner's name alone.
 */
export const getGalleryBoardOwnerLabel = (board: GalleryBoard, t: Translate): string | null => {
  const { access } = board;
  if (access === undefined) {
    return board.ownerName ?? null;
  }
  if (access.role === 'owner') {
    return null;
  }
  const owner = board.ownerName ?? t('widgets.gallery.boardAccess.deletedOwner');
  const level =
    access.adminOverride || access.role === null
      ? t('widgets.gallery.boardAccess.admin')
      : t(`widgets.gallery.boardAccess.${access.role}`);
  return t('widgets.gallery.boardAccess.ownerAndAccess', { access: level, owner });
};
