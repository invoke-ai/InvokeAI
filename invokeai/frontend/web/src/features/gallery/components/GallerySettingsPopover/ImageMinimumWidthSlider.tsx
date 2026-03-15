import { CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectGalleryColumns } from 'features/gallery/store/gallerySelectors';
import { setGalleryColumns } from 'features/gallery/store/gallerySlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const GalleryColumnsSlider = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const galleryColumns = useAppSelector(selectGalleryColumns);

  const onChange = useCallback(
    (v: number) => {
      dispatch(setGalleryColumns(v));
    },
    [dispatch]
  );
  return (
    <FormControl>
      <FormLabel>{t('gallery.columns')}</FormLabel>
      <CompositeSlider value={galleryColumns} onChange={onChange} min={2} max={15} defaultValue={5} />
    </FormControl>
  );
};

export default memo(GalleryColumnsSlider);
