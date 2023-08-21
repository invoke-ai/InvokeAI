import IAIIconButton from 'common/components/IAIIconButton';
import { memo, useCallback } from 'react';
import { FaPlus } from 'react-icons/fa';
import { useCreateBoardMutation } from 'services/api/endpoints/boards';

const DEFAULT_BOARD_NAME = 'My Board';

const AddBoardButton = () => {
  const [createBoard, { isLoading }] = useCreateBoardMutation();

  const handleCreateBoard = useCallback(() => {
    createBoard(DEFAULT_BOARD_NAME);
  }, [createBoard]);

  return (
    <IAIIconButton
      icon={<FaPlus />}
      isLoading={isLoading}
      tooltip="Add Board"
      aria-label="Add Board"
      onClick={handleCreateBoard}
      size="sm"
    />
  );
};

export default memo(AddBoardButton);
