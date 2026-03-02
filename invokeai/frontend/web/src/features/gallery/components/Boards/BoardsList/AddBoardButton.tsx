import { Button, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { boardIdSelected, boardSearchTextChanged } from 'features/gallery/store/gallerySlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';
import { useCreateBoardMutation } from 'services/api/endpoints/boards';

const AddBoardButton = ({ variant = 'default' }: { variant?: 'icon' | 'default' }) => {
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

  return variant === 'icon' ? (
    <IconButton
      aria-label={t('boards.addBoard')}
      tooltip={t('boards.addBoard')}
      icon={<PiPlusBold />}
      isLoading={isLoading}
      onClick={handleCreateBoard}
      size="sm"
      data-testid="add-board-icon-button"
      variant="ghost"
    />
  ) : (
    <Button
      leftIcon={<PiPlusBold />}
      isLoading={isLoading}
      onClick={handleCreateBoard}
      size="sm"
      data-testid="add-board-button"
      variant="ghost"
      flex={1}
      justifyContent="start"
    >
      {t('boards.addBoard')}
    </Button>
  );
};

export default memo(AddBoardButton);
