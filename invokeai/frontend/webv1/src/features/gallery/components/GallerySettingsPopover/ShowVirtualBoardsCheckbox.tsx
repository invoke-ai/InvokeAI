import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectGallerySlice, showVirtualBoardsChanged } from 'features/gallery/store/gallerySlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';

const selectShowVirtualBoards = createSelector(selectGallerySlice, (gallery) => gallery.showVirtualBoards);

const ShowVirtualBoardsCheckbox = () => {
  const dispatch = useAppDispatch();
  const showVirtualBoards = useAppSelector(selectShowVirtualBoards);

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(showVirtualBoardsChanged(e.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <FormLabel flexGrow={1}>Virtual Boards</FormLabel>
      <Checkbox isChecked={showVirtualBoards} onChange={onChange} />
    </FormControl>
  );
};

export default memo(ShowVirtualBoardsCheckbox);
