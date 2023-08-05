import { useAppSelector } from 'app/store/storeHooks';
import { BoardId } from 'features/gallery/store/types';
import { useMemo } from 'react';
import {
  useGetBoardAssetsTotalQuery,
  useGetBoardImagesTotalQuery,
} from '../endpoints/boards';

export const useBoardTotal = (board_id: BoardId) => {
  const galleryView = useAppSelector((state) => state.gallery.galleryView);

  const { data: totalImages } = useGetBoardImagesTotalQuery(board_id);
  const { data: totalAssets } = useGetBoardAssetsTotalQuery(board_id);

  const currentViewTotal = useMemo(
    () => (galleryView === 'images' ? totalImages : totalAssets),
    [galleryView, totalAssets, totalImages]
  );

  return { totalImages, totalAssets, currentViewTotal };
};
