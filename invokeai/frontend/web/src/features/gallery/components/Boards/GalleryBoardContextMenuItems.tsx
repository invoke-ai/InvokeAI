import { MenuItem } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { autoAddBoardIdChanged } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useMemo } from 'react';
import { FaMinus, FaPlus, FaTrash } from 'react-icons/fa';
import { BoardDTO } from 'services/api/types';

type Props = {
  board: BoardDTO;
  setBoardToDelete?: (board?: BoardDTO) => void;
};

const GalleryBoardContextMenuItems = ({ board, setBoardToDelete }: Props) => {
  const dispatch = useAppDispatch();

  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ gallery }) => {
          const isSelectedForAutoAdd =
            board.board_id === gallery.autoAddBoardId;

          return { isSelectedForAutoAdd };
        },
        defaultSelectorOptions
      ),
    [board.board_id]
  );

  const { isSelectedForAutoAdd } = useAppSelector(selector);

  const handleDelete = useCallback(() => {
    if (!setBoardToDelete) {
      return;
    }
    setBoardToDelete(board);
  }, [board, setBoardToDelete]);

  const handleToggleAutoAdd = useCallback(() => {
    dispatch(
      autoAddBoardIdChanged(isSelectedForAutoAdd ? null : board.board_id)
    );
  }, [board.board_id, dispatch, isSelectedForAutoAdd]);

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
        icon={isSelectedForAutoAdd ? <FaMinus /> : <FaPlus />}
        onClickCapture={handleToggleAutoAdd}
      >
        {isSelectedForAutoAdd ? 'Disable Auto-Add' : 'Auto-Add to this Board'}
      </MenuItem>
      <MenuItem
        sx={{ color: 'error.600', _dark: { color: 'error.300' } }}
        icon={<FaTrash />}
        onClickCapture={handleDelete}
      >
        Delete Board
      </MenuItem>
    </>
  );
};

export default memo(GalleryBoardContextMenuItems);
