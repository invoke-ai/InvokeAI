import IAIIconButton from 'common/components/IAIIconButton';
import { memo, useCallback } from 'react';
import { FaPlus } from 'react-icons/fa';
import { useCreateBoardMutation } from 'services/api/endpoints/boards';
import { useTranslation } from 'react-i18next';

const AddBoardButton = () => {
  const { t } = useTranslation();
  const [createBoard, { isLoading }] = useCreateBoardMutation();
  const DEFAULT_BOARD_NAME = t('boards.myBoard');
  const handleCreateBoard = useCallback(() => {
    createBoard(DEFAULT_BOARD_NAME);
  }, [createBoard, DEFAULT_BOARD_NAME]);

  return (
    <IAIIconButton
      icon={<FaPlus />}
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
