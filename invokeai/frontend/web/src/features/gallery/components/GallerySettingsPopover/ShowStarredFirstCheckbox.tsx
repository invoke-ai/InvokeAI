import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectGallerySlice, starredFirstChanged } from 'features/gallery/store/gallerySlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectStarredFirst = createSelector(selectGallerySlice, (gallery) => gallery.starredFirst);

const GallerySettingsPopover = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const starredFirst = useAppSelector(selectStarredFirst);

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(starredFirstChanged(e.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl w="full">
      <FormLabel flexGrow={1} m={0}>
        {t('gallery.showStarredImagesFirst')}
      </FormLabel>
      <Switch size="sm" isChecked={starredFirst} onChange={onChange} />
    </FormControl>
  );
};

export default memo(GallerySettingsPopover);
