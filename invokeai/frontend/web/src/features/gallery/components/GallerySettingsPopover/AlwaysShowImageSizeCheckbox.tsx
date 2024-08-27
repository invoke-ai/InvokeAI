import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { alwaysShowImageSizeBadgeChanged, selectGallerySlice } from 'features/gallery/store/gallerySlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectAlwaysShowImageSizeBadge = createSelector(
  selectGallerySlice,
  (gallery) => gallery.alwaysShowImageSizeBadge
);

const GallerySettingsPopover = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const alwaysShowImageSizeBadge = useAppSelector(selectAlwaysShowImageSizeBadge);

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(alwaysShowImageSizeBadgeChanged(e.target.checked)),
    [dispatch]
  );

  return (
    <FormControl>
      <FormLabel flexGrow={1}>{t('gallery.alwaysShowImageSizeBadge')}</FormLabel>
      <Checkbox isChecked={alwaysShowImageSizeBadge} onChange={onChange} />
    </FormControl>
  );
};

export default memo(GallerySettingsPopover);
