import { useAppDispatch } from 'app/store/storeHooks';
import { IconMenuItem } from 'common/components/IconMenuItem';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { imageOpenedInNewTab } from 'features/gallery/store/actions';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowSquareOutBold } from 'react-icons/pi';

export const ImageMenuItemOpenInNewTab = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useImageDTOContext();
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    window.open(imageDTO.image_url, '_blank');
    dispatch(imageOpenedInNewTab());
  }, [imageDTO.image_url, dispatch]);

  return (
    <IconMenuItem
      onClickCapture={onClick}
      aria-label={t('common.openInNewTab')}
      tooltip={t('common.openInNewTab')}
      icon={<PiArrowSquareOutBold />}
    />
  );
});

ImageMenuItemOpenInNewTab.displayName = 'ImageMenuItemOpenInNewTab';
