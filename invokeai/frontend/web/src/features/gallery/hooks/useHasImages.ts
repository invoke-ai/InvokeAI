import { useMemo } from 'react';
import { useGetBoardVideosTotalQuery, useListAllBoardsQuery } from 'services/api/endpoints/boards';
import { useListImagesQuery } from 'services/api/endpoints/images';

export const LOADING_SYMBOL = Symbol('LOADING');

/**
 * Decides whether the gallery has any content. A gallery containing only uncategorized
 * videos must count as non-empty — the video totals are not derivable from the boards
 * list (which only covers real boards) or the uncategorized *image* total. Exported for
 * tests.
 */
export const getHasGalleryContent = (
  boardList: { image_count: number; video_count?: number }[] | undefined,
  uncategorizedImagesTotal: number | undefined,
  uncategorizedVideosTotal: number | undefined
): boolean => {
  if (boardList && boardList.some((board) => board.image_count + (board.video_count ?? 0) > 0)) {
    return true;
  }
  if (uncategorizedImagesTotal === undefined && uncategorizedVideosTotal === undefined) {
    // No data at all — default to "has content" so we don't flash the new-user view.
    return true;
  }
  return (uncategorizedImagesTotal ?? 0) > 0 || (uncategorizedVideosTotal ?? 0) > 0;
};

export const useHasImages = () => {
  const { data: boardList, isLoading: loadingBoards } = useListAllBoardsQuery({ include_archived: true });
  const { data: uncategorizedImages, isLoading: loadingImages } = useListImagesQuery({
    board_id: 'none',
    offset: 0,
    limit: 0,
    is_intermediate: false,
  });
  const { data: uncategorizedVideos, isLoading: loadingVideos } = useGetBoardVideosTotalQuery('none');

  const hasImages = useMemo(() => {
    // default to true
    if (loadingBoards || loadingImages || loadingVideos) {
      return LOADING_SYMBOL;
    }

    return getHasGalleryContent(boardList, uncategorizedImages?.total, uncategorizedVideos?.total);
  }, [boardList, uncategorizedImages, uncategorizedVideos, loadingBoards, loadingImages, loadingVideos]);

  return hasImages;
};
