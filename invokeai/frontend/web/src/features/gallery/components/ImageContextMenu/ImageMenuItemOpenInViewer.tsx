import { IconButton } from '@invoke-ai/ui-library';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsOutBold } from 'react-icons/pi';

export const ImageMenuItemOpenInViewer = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useImageDTOContext();
  const imageViewer = useImageViewer();
  const onPointerUp = useCallback(() => {
    imageViewer.openImageInViewer(imageDTO);
  }, [imageDTO, imageViewer]);

  return (
    <IconButton
      icon={<PiArrowsOutBold />}
      onPointerUp={onPointerUp}
      aria-label={t('common.openInViewer')}
      tooltip={t('common.openInViewer')}
      variant="ghost"
      colorScheme="base"
    />
  );
});

ImageMenuItemOpenInViewer.displayName = 'ImageMenuItemOpenInViewer';
