import { BoardId } from 'features/gallery/store/gallerySlice';
import { useListAllBoardsQuery } from '../endpoints/boards';

export const useBoardName = (board_id: BoardId | null | undefined) => {
  const { boardName } = useListAllBoardsQuery(undefined, {
    selectFromResult: ({ data }) => {
      let boardName = '';
      if (board_id === 'images') {
        boardName = 'All Images';
      } else if (board_id === 'assets') {
        boardName = 'All Assets';
      } else if (board_id === 'no_board') {
        boardName = 'No Board';
      } else if (board_id === 'batch') {
        boardName = 'Batch';
      } else {
        const selectedBoard = data?.find((b) => b.board_id === board_id);
        boardName = selectedBoard?.board_name || 'Unknown Board';
      }

      return { boardName };
    },
  });

  return boardName;
};
