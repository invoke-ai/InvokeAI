import { IconButton } from '@invoke-ai/ui-library';
import type { AppThunkDispatch } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectAutoAssignBoardOnClick } from 'features/gallery/store/gallerySelectors';
import { autoAddBoardIdChanged, boardIdSelected, boardSearchTextChanged } from 'features/gallery/store/gallerySlice';
import type { BoardId } from 'features/gallery/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';
import { useCreateBoardMutation } from 'services/api/endpoints/boards';

export const getCreatedBoardActions = (boardId: BoardId, autoAssignBoardOnClick: boolean) => [
  boardIdSelected({ boardId }),
  ...(autoAssignBoardOnClick ? [autoAddBoardIdChanged(boardId)] : []),
  boardSearchTextChanged(''),
];

type CreateBoard = (args: { board_name: string }) => { unwrap: () => Promise<{ board_id: BoardId }> };

export const createBoardAndDispatchActions = async (
  createBoard: CreateBoard,
  dispatch: AppThunkDispatch,
  boardName: string,
  autoAssignBoardOnClick: boolean
) => {
  try {
    const board = await createBoard({ board_name: boardName }).unwrap();
    getCreatedBoardActions(board.board_id, autoAssignBoardOnClick).forEach((action) => dispatch(action));
  } catch {
    //no-op
  }
};

const AddBoardButton = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const autoAssignBoardOnClick = useAppSelector(selectAutoAssignBoardOnClick);
  const [createBoard, { isLoading }] = useCreateBoardMutation();

  const handleCreateBoard = useCallback(
    () => createBoardAndDispatchActions(createBoard, dispatch, t('boards.myBoard'), autoAssignBoardOnClick),
    [t, createBoard, dispatch, autoAssignBoardOnClick]
  );

  return (
    <IconButton
      icon={<PiPlusBold />}
      isLoading={isLoading}
      tooltip={t('boards.addBoard')}
      aria-label={t('boards.addBoard')}
      onClick={handleCreateBoard}
      size="md"
      data-testid="add-board-button"
      variant="link"
      w={8}
      h={8}
    />
  );
};

export default memo(AddBoardButton);
