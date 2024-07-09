import { IconButton } from '@invoke-ai/ui-library';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';
import { useCreateBoardMutation } from 'services/api/endpoints/boards';

type Props = {
  isPrivateBoard: boolean;
};

const AddBoardButton = ({ isPrivateBoard }: Props) => {
  const { t } = useTranslation();
  const [createBoard, { isLoading }] = useCreateBoardMutation();
  const DEFAULT_BOARD_NAME = t('boards.myBoard');
  const handleCreateBoard = useCallback(() => {
    createBoard({ board_name: DEFAULT_BOARD_NAME, is_private: isPrivateBoard });
  }, [createBoard, DEFAULT_BOARD_NAME, isPrivateBoard]);

  return (
    <IconButton
      icon={<PiPlusBold />}
      isLoading={isLoading}
      tooltip={t('boards.addBoard')}
      aria-label={t('boards.addBoard')}
      onClick={handleCreateBoard}
      size="md"
      data-testid="add-board-button"
      variant="ghost"
    />
  );
};

export default memo(AddBoardButton);
