import { useMemo } from 'react';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';
import { useListImagesQuery } from 'services/api/endpoints/images';

export const LOADING_SYMBOL = Symbol('LOADING');

export const useHasImages = () => {
  const { data: boardList, isLoading: loadingBoards } = useListAllBoardsQuery({ include_archived: true });
  const { data: uncategorizedImages, isLoading: loadingImages } = useListImagesQuery({
    board_id: 'none',
    offset: 0,
    limit: 0,
    is_intermediate: false,
  });

  const hasImages = useMemo(() => {
    // default to true
    if (loadingBoards || loadingImages) {
      return LOADING_SYMBOL;
    }

    const hasBoards = boardList && boardList.length > 0;

    if (hasBoards) {
      if (boardList.filter((board) => board.image_count > 0).length > 0) {
        return true;
      }
    }
    return uncategorizedImages ? uncategorizedImages.total > 0 : true;
  }, [boardList, uncategorizedImages, loadingBoards, loadingImages]);

  return hasImages;
};
