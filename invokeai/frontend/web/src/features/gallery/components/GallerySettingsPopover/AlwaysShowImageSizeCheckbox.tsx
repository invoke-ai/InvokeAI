import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { alwaysShowImageSizeBadgeChanged } from 'features/gallery/store/gallerySlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const GallerySettingsPopover = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const alwaysShowImageSizeBadge = useAppSelector((s) => s.gallery.alwaysShowImageSizeBadge);

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
