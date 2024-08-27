import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectGallerySlice, shouldAutoSwitchChanged } from 'features/gallery/store/gallerySlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectShouldAutoSwitch = createSelector(selectGallerySlice, (gallery) => gallery.shouldAutoSwitch);

const GallerySettingsPopover = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const shouldAutoSwitch = useAppSelector(selectShouldAutoSwitch);

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(shouldAutoSwitchChanged(e.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <FormLabel flexGrow={1}>{t('gallery.autoSwitchNewImages')}</FormLabel>
      <Checkbox isChecked={shouldAutoSwitch} onChange={onChange} />
    </FormControl>
  );
};

export default memo(GallerySettingsPopover);
