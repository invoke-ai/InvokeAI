import { MenuItem } from '@chakra-ui/react';
import { useCallback } from 'react';
import { FaFolder, FaFolderPlus, FaTrash } from 'react-icons/fa';

const MultipleSelectionMenuItems = () => {
  const handleAddSelectionToBoard = useCallback(() => {
    // TODO: add selection to board
  }, []);

  const handleDeleteSelection = useCallback(() => {
    // TODO: delete all selected images
  }, []);

  const handleAddSelectionToBatch = useCallback(() => {
    // TODO: add selection to batch
  }, []);

  return (
    <>
      <MenuItem icon={<FaFolder />} onClickCapture={handleAddSelectionToBoard}>
        Move Selection to Board
      </MenuItem>
      <MenuItem
        icon={<FaFolderPlus />}
        onClickCapture={handleAddSelectionToBatch}
      >
        Add Selection to Batch
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
