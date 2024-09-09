import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { imageSelected, imageToCompareChanged } from 'features/gallery/store/gallerySlice';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEyeBold } from 'react-icons/pi';

export const ImageMenuItemOpenInViewer = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const imageDTO = useImageDTOContext();

  const onClick = useCallback(() => {
    dispatch(imageToCompareChanged(null));
    dispatch(imageSelected(imageDTO));
    dispatch(setActiveTab('gallery'));
  }, [dispatch, imageDTO]);

  return (
    <MenuItem icon={<PiEyeBold />} onClick={onClick}>
      {t('gallery.openInViewer')}
    </MenuItem>
  );
});

ImageMenuItemOpenInViewer.displayName = 'ImageMenuItemOpenInViewer';
