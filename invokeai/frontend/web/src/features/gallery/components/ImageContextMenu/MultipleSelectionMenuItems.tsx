import { MenuItem } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  imagesToChangeSelected,
  isModalOpenChanged,
} from 'features/changeBoardModal/store/slice';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import { useCallback } from 'react';
import { FaFolder, FaTrash } from 'react-icons/fa';

const MultipleSelectionMenuItems = () => {
  const dispatch = useAppDispatch();
  const selection = useAppSelector((state) => state.gallery.selection);

  const handleChangeBoard = useCallback(() => {
    dispatch(imagesToChangeSelected(selection));
    dispatch(isModalOpenChanged(true));
  }, [dispatch, selection]);

  const handleDeleteSelection = useCallback(() => {
    dispatch(imagesToDeleteSelected(selection));
  }, [dispatch, selection]);

  return (
    <>
      <MenuItem icon={<FaFolder />} onClickCapture={handleChangeBoard}>
        Change Board
      </MenuItem>
      <MenuItem
        sx={{ color: 'error.600', _dark: { color: 'error.300' } }}
        icon={<FaTrash />}
        onClickCapture={handleDeleteSelection}
      >
        Delete Selection
      </MenuItem>
    </>
  );
};

export default MultipleSelectionMenuItems;
