import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectAutoAssignBoardOnClick } from 'features/gallery/store/gallerySelectors';
import { autoAssignBoardOnClickChanged } from 'features/gallery/store/gallerySlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const GallerySettingsPopover = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const autoAssignBoardOnClick = useAppSelector(selectAutoAssignBoardOnClick);

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(autoAssignBoardOnClickChanged(e.target.checked)),
    [dispatch]
  );

  return (
    <FormControl>
      <FormLabel flexGrow={1}>{t('gallery.autoAssignBoardOnClick')}</FormLabel>
      <Checkbox isChecked={autoAssignBoardOnClick} onChange={onChange} />
    </FormControl>
  );
};

export default memo(GallerySettingsPopover);
