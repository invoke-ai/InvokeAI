import { useAppSelector } from 'app/store/storeHooks';
import type { BoardId } from 'features/gallery/store/types';
import { useMemo } from 'react';
import { useGetBoardAssetsTotalQuery, useGetBoardImagesTotalQuery } from 'services/api/endpoints/boards';

export const useBoardTotal = (board_id: BoardId) => {
  const galleryView = useAppSelector((s) => s.gallery.galleryView);

  const { data: totalImages } = useGetBoardImagesTotalQuery(board_id);
  const { data: totalAssets } = useGetBoardAssetsTotalQuery(board_id);

  const currentViewTotal = useMemo(
    () => (galleryView === 'images' ? totalImages?.total : totalAssets?.total),
    [galleryView, totalAssets, totalImages]
  );

  return { totalImages, totalAssets, currentViewTotal };
};
