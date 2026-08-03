import type { GalleryBoard, GalleryView } from '@features/gallery/core/types';

import { getBoardCounts } from './galleryStateView';

/**
 * How many items the active tab would show for a board, so a count always
 * answers "how many of what I'm looking at".
 */
export const getGalleryCountForView = (board: GalleryBoard, galleryView: GalleryView): number => {
  const counts = getBoardCounts(board);

  return galleryView === 'assets' ? counts.assetCount : counts.imageCount + counts.videoCount;
};
