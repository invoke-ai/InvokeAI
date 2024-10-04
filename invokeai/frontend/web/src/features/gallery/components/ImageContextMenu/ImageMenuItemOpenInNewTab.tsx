import { IconButton } from '@invoke-ai/ui-library';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowSquareOutBold } from 'react-icons/pi';

export const ImageMenuItemOpenInNewTab = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useImageDTOContext();
  const onPointerUp = useCallback(() => {
    window.open(imageDTO.image_url, '_blank');
  }, [imageDTO.image_url]);

  return (
    <IconButton
      onPointerUpCapture={onPointerUp}
      aria-label={t('common.openInNewTab')}
      tooltip={t('common.openInNewTab')}
      icon={<PiArrowSquareOutBold />}
      variant="ghost"
      colorScheme="base"
    />
  );
});

ImageMenuItemOpenInNewTab.displayName = 'ImageMenuItemOpenInNewTab';
