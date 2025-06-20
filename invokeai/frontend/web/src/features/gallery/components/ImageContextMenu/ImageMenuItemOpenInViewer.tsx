import { useAppDispatch } from 'app/store/storeHooks';
import { IconMenuItem } from 'common/components/IconMenuItem';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { imageSelected, imageToCompareChanged } from 'features/gallery/store/gallerySlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsOutBold } from 'react-icons/pi';

export const ImageMenuItemOpenInViewer = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const imageDTO = useImageDTOContext();
  const onClick = useCallback(() => {
    dispatch(imageToCompareChanged(null));
    dispatch(imageSelected(imageDTO));
    // TODO: figure out how to select the closest image viewer...
  }, [dispatch, imageDTO]);

  return (
    <IconMenuItem
      icon={<PiArrowsOutBold />}
      onClickCapture={onClick}
      aria-label={t('common.openInViewer')}
      tooltip={t('common.openInViewer')}
    />
  );
});

ImageMenuItemOpenInViewer.displayName = 'ImageMenuItemOpenInViewer';
