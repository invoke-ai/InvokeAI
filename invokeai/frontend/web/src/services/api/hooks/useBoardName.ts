import { BoardId } from 'features/gallery/store/types';
import { useListAllBoardsQuery } from '../endpoints/boards';
import { t } from 'i18next';

export const useBoardName = (board_id: BoardId) => {
  const { boardName } = useListAllBoardsQuery(undefined, {
    selectFromResult: ({ data }) => {
      const selectedBoard = data?.find((b) => b.board_id === board_id);
      const boardName = selectedBoard?.board_name || t('boards.uncategorized');

      return { boardName };
    },
  });

  return boardName;
};
