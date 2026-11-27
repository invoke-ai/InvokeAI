import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectShouldUsePagedGalleryView } from 'features/ui/store/uiSelectors';
import { setShouldUsePagedGalleryView } from 'features/ui/store/uiSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const UsePagedGalleryViewCheckbox = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const shouldUsePagedGalleryView = useAppSelector(selectShouldUsePagedGalleryView);

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setShouldUsePagedGalleryView(e.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <FormLabel flexGrow={1}>{t('gallery.usePagedGalleryView')}</FormLabel>
      <Checkbox isChecked={shouldUsePagedGalleryView} onChange={onChange} />
    </FormControl>
  );
};

export default memo(UsePagedGalleryViewCheckbox);
