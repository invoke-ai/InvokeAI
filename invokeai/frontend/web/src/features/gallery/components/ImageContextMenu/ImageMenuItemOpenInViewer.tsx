import { IconMenuItem } from 'common/components/IconMenuItem';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsOutBold } from 'react-icons/pi';

export const ImageMenuItemOpenInViewer = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useImageDTOContext();
  const imageViewer = useImageViewer();
  const onClick = useCallback(() => {
    imageViewer.openImageInViewer(imageDTO);
  }, [imageDTO, imageViewer]);

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
