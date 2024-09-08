import { CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectGalleryImageMinimumWidth } from 'features/gallery/store/gallerySelectors';
import { setGalleryImageMinimumWidth } from 'features/gallery/store/gallerySlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const GallerySettingsPopover = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const galleryImageMinimumWidth = useAppSelector(selectGalleryImageMinimumWidth);

  const onChange = useCallback(
    (v: number) => {
      dispatch(setGalleryImageMinimumWidth(v));
    },
    [dispatch]
  );
  return (
    <FormControl>
      <FormLabel>{t('gallery.galleryImageSize')}</FormLabel>
      <CompositeSlider value={galleryImageMinimumWidth} onChange={onChange} min={45} max={256} defaultValue={90} />
    </FormControl>
  );
};

export default memo(GallerySettingsPopover);
