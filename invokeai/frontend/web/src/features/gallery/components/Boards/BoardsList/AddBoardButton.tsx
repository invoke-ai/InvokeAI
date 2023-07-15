import IAIButton from 'common/components/IAIButton';
import { useCallback } from 'react';
import { useCreateBoardMutation } from 'services/api/endpoints/boards';

const DEFAULT_BOARD_NAME = 'My Board';

const AddBoardButton = () => {
  const [createBoard, { isLoading }] = useCreateBoardMutation();

  const handleCreateBoard = useCallback(() => {
    createBoard(DEFAULT_BOARD_NAME);
  }, [createBoard]);

  return (
    <IAIButton
      isLoading={isLoading}
      aria-label="Add Board"
      onClick={handleCreateBoard}
      size="sm"
      sx={{ px: 4 }}
    >
      Add Board
    </IAIButton>
  );
};

export default AddBoardButton;
