import { useAppSelector } from 'app/store/storeHooks';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';

/**
 * Returns the `BoardDTO` for the board currently configured as the auto-add
 * destination, or `null` when it is set to "Uncategorized" (`boardId === 'none'`)
 * or when the board list has not yet loaded.
 */
export const useAutoAddBoard = () => {
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);
  const { board } = useListAllBoardsQuery(
    { include_archived: true },
    {
      selectFromResult: ({ data }) => ({
        board: data?.find((b) => b.board_id === autoAddBoardId) ?? null,
      }),
    }
  );
  return board;
};
