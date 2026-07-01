import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectGallerySlice, shouldShowArchivedBoardsChanged } from 'features/gallery/store/gallerySlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectShouldShowArchivedBoards = createSelector(
  selectGallerySlice,
  (gallery) => gallery.shouldShowArchivedBoards
);

const GallerySettingsPopover = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const shouldShowArchivedBoards = useAppSelector(selectShouldShowArchivedBoards);

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(shouldShowArchivedBoardsChanged(e.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <FormLabel flexGrow={1}>{t('gallery.showArchivedBoards')}</FormLabel>
      <Checkbox isChecked={shouldShowArchivedBoards} onChange={onChange} />
    </FormControl>
  );
};

export default memo(GallerySettingsPopover);
