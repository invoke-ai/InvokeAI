import { MenuItem } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { autoAddBoardIdChanged } from 'features/gallery/store/gallerySlice';
import { memo, useCallback } from 'react';
import { FaPlus } from 'react-icons/fa';

const NoBoardContextMenuItems = () => {
  const dispatch = useAppDispatch();

  const autoAddBoardId = useAppSelector(
    (state) => state.gallery.autoAddBoardId
  );
  const handleDisableAutoAdd = useCallback(() => {
    dispatch(autoAddBoardIdChanged(undefined));
  }, [dispatch]);

  return (
    <>
      {autoAddBoardId && (
        <MenuItem icon={<FaPlus />} onClickCapture={handleDisableAutoAdd}>
          Disable Auto-Add
        </MenuItem>
      )}
    </>
  );
};

export default memo(NoBoardContextMenuItems);
