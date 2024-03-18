import { IconButton } from '@invoke-ai/ui-library';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';
import { useCreateBoardMutation } from 'services/api/endpoints/boards';

const AddBoardButton = () => {
  const { t } = useTranslation();
  const [createBoard, { isLoading }] = useCreateBoardMutation();
  const DEFAULT_BOARD_NAME = t('boards.myBoard');
  const handleCreateBoard = useCallback(() => {
    createBoard(DEFAULT_BOARD_NAME);
  }, [createBoard, DEFAULT_BOARD_NAME]);

  return (
    <IconButton
      icon={<PiPlusBold />}
      isLoading={isLoading}
      tooltip={t('boards.addBoard')}
      aria-label={t('boards.addBoard')}
      onClick={handleCreateBoard}
      size="sm"
      data-testid="add-board-button"
    />
  );
};

export default memo(AddBoardButton);
