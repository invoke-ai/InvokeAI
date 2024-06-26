import type { BoardId } from 'features/gallery/store/types';
import { t } from 'i18next';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';
import { selectListBoardsQueryArgs } from '../../../features/gallery/store/gallerySelectors';
import { useAppSelector } from '../../../app/store/storeHooks';

export const useBoardName = (board_id: BoardId) => {
  const queryArgs = useAppSelector(selectListBoardsQueryArgs);
  const { boardName } = useListAllBoardsQuery(queryArgs, {
    selectFromResult: ({ data }) => {
      const selectedBoard = data?.find((b) => b.board_id === board_id);
      const boardName = selectedBoard?.board_name || t('boards.uncategorized');

      return { boardName };
    },
  });

  return boardName;
};
