import IAIIconButton from 'common/components/IAIIconButton';
import { useCallback } from 'react';
import { FaPlus } from 'react-icons/fa';
import { useCreateBoardMutation } from 'services/apiSlice';

const DEFAULT_BOARD_NAME = 'My Board';

const AddBoardButton = () => {
  const [createBoard, { isLoading }] = useCreateBoardMutation();

  const handleCreateBoard = useCallback(() => {
    createBoard(DEFAULT_BOARD_NAME);
  }, [createBoard]);

  return (
    <IAIIconButton
      tooltip="Add Board"
      isLoading={isLoading}
      aria-label="Add Board"
      onClick={handleCreateBoard}
      size="xs"
      icon={<FaPlus />}
      sx={{ p: 2 }}
    />
  );
};

export default AddBoardButton;
