import { MenuItem } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  imagesToChangeSelected,
  isModalOpenChanged,
} from 'features/changeBoardModal/store/slice';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import { memo, useCallback, useMemo } from 'react';
import { FaFolder, FaTrash } from 'react-icons/fa';
import { MdStar, MdStarBorder } from 'react-icons/md';
import {
  useStarImagesMutation,
  useUnstarImagesMutation,
} from '../../../../services/api/endpoints/images';

const MultipleSelectionMenuItems = () => {
  const dispatch = useAppDispatch();
  const selection = useAppSelector((state) => state.gallery.selection);

  const [starImages] = useStarImagesMutation();
  const [unstarImages] = useUnstarImagesMutation();

  const handleChangeBoard = useCallback(() => {
    dispatch(imagesToChangeSelected(selection));
    dispatch(isModalOpenChanged(true));
  }, [dispatch, selection]);

  const handleDeleteSelection = useCallback(() => {
    dispatch(imagesToDeleteSelected(selection));
  }, [dispatch, selection]);

  const handleStarSelection = useCallback(() => {
    starImages({ imageDTOs: selection });
  }, [starImages, selection]);

  const handleUnstarSelection = useCallback(() => {
    unstarImages({ imageDTOs: selection });
  }, [unstarImages, selection]);

  const areAllStarred = useMemo(() => {
    return selection.every((img) => img.starred);
  }, [selection]);

  const areAllUnstarred = useMemo(() => {
    return selection.every((img) => !img.starred);
  }, [selection]);

  return (
    <>
      {areAllStarred && (
        <MenuItem
          icon={<MdStarBorder />}
          onClickCapture={handleUnstarSelection}
        >
          Unstar All
        </MenuItem>
      )}
      {(areAllUnstarred || (!areAllStarred && !areAllUnstarred)) && (
        <MenuItem icon={<MdStar />} onClickCapture={handleStarSelection}>
          Star All
        </MenuItem>
      )}
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

export default memo(MultipleSelectionMenuItems);
