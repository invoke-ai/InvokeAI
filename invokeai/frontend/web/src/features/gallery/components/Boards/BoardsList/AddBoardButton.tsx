import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { boardIdSelected, boardSearchTextChanged } from 'features/gallery/store/gallerySlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';
import { useCreateBoardMutation } from 'services/api/endpoints/boards';

const AddBoardButton = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const [createBoard, { isLoading }] = useCreateBoardMutation();

  const handleCreateBoard = useCallback(async () => {
    try {
      const board = await createBoard({ board_name: t('boards.myBoard') }).unwrap();
      dispatch(boardIdSelected({ boardId: board.board_id }));
      dispatch(boardSearchTextChanged(''));
    } catch {
      //no-op
    }
  }, [t, createBoard, dispatch]);

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
