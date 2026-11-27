import { useAppSelector } from 'app/store/storeHooks';
import { selectSelectedBoardId } from 'features/gallery/store/gallerySelectors';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';

/**
 * Returns the `BoardDTO` for the currently selected board, or `null` when the
 * user is viewing "Uncategorized" (`boardId === 'none'`) or when the board list
 * has not yet loaded.
 */
export const useSelectedBoard = () => {
  const selectedBoardId = useAppSelector(selectSelectedBoardId);
  const { board } = useListAllBoardsQuery(
    { include_archived: true },
    {
      selectFromResult: ({ data }) => ({
        board: data?.find((b) => b.board_id === selectedBoardId) ?? null,
      }),
    }
  );
  return board;
};
