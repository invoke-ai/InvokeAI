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
