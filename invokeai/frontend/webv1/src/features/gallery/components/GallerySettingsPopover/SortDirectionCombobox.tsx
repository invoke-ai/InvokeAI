import type { ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { SingleValue } from 'chakra-react-select';
import { orderDirChanged, selectGallerySlice } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { assert } from 'tsafe';

const selectOrderDir = createSelector(selectGallerySlice, (gallery) => gallery.orderDir);

const GallerySettingsPopover = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const orderDir = useAppSelector(selectOrderDir);

  const options = useMemo<ComboboxOption[]>(
    () => [
      { value: 'DESC', label: t('gallery.newestFirst') },
      { value: 'ASC', label: t('gallery.oldestFirst') },
    ],
    [t]
  );

  const onChange = useCallback(
    (v: SingleValue<ComboboxOption>) => {
      assert(v?.value === 'ASC' || v?.value === 'DESC');
      dispatch(orderDirChanged(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => {
    return options.find((opt) => opt.value === orderDir);
  }, [orderDir, options]);

  return (
    <FormControl>
      <FormLabel flexGrow={1} m={0}>
        {t('gallery.sortDirection')}
      </FormLabel>
      <Combobox isSearchable={false} value={value} options={options} onChange={onChange} />
    </FormControl>
  );
};

export default memo(GallerySettingsPopover);
