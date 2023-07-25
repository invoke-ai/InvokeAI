import { MenuItem } from '@chakra-ui/react';
import { memo, useCallback } from 'react';
import { FaTrash } from 'react-icons/fa';
import { BoardDTO } from 'services/api/types';

type Props = {
  board: BoardDTO;
  setBoardToDelete?: (board?: BoardDTO) => void;
};

const GalleryBoardContextMenuItems = ({ board, setBoardToDelete }: Props) => {
  const handleDelete = useCallback(() => {
    if (!setBoardToDelete) {
      return;
    }
    setBoardToDelete(board);
  }, [board, setBoardToDelete]);

  return (
    <>
      {board.image_count > 0 && (
        <>
          {/* <MenuItem
                    isDisabled={!board.image_count}
                    icon={<FaImages />}
                    onClickCapture={handleAddBoardToBatch}
                  >
                    Add Board to Batch
                  </MenuItem> */}
        </>
      )}
      <MenuItem
        sx={{ color: 'error.600', _dark: { color: 'error.300' } }}
        icon={<FaTrash />}
        onClick={handleDelete}
      >
        Delete Board
      </MenuItem>
    </>
  );
};

export default memo(GalleryBoardContextMenuItems);
