import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { starredFirstChanged } from 'features/gallery/store/gallerySlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const GallerySettingsPopover = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const starredFirst = useAppSelector((s) => s.gallery.starredFirst);

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
