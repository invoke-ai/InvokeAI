import type { IconButtonProps } from '@invoke-ai/ui-library';
import { Button, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { boardIdSelected, boardSearchTextChanged } from 'features/gallery/store/gallerySlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';
import { useCreateBoardMutation } from 'services/api/endpoints/boards';

const useAddBoard = () => {
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

  return { handleCreateBoard, isLoading, t };
};

export const AddBoardButton = memo(() => {
  const { handleCreateBoard, isLoading, t } = useAddBoard();

  return (
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
});

AddBoardButton.displayName = 'AddBoardButton';

export const AddBoardIconButton = memo((props: Partial<IconButtonProps>) => {
  const { handleCreateBoard, isLoading, t } = useAddBoard();
  const { 'aria-label': ariaLabel = t('boards.addBoard'), ...rest } = props;

  return (
    <IconButton
      aria-label={ariaLabel}
      tooltip={t('boards.addBoard')}
      icon={<PiPlusBold />}
      isLoading={isLoading}
      onClick={handleCreateBoard}
      data-testid="add-board-icon-button"
      variant="ghost"
      {...rest}
    />
  );
});

AddBoardIconButton.displayName = 'AddBoardIconButton';
