import { IconButton } from '@invoke-ai/ui-library';
import { useDownloadImage } from 'common/hooks/useDownloadImage';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadSimpleBold } from 'react-icons/pi';

export const ImageMenuItemDownload = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useImageDTOContext();
  const { downloadImage } = useDownloadImage();

  const onPointerUp = useCallback(() => {
    downloadImage(imageDTO.image_url, imageDTO.image_name);
  }, [downloadImage, imageDTO.image_name, imageDTO.image_url]);

  return (
    <IconButton
      icon={<PiDownloadSimpleBold />}
      aria-label={t('parameters.downloadImage')}
      tooltip={t('parameters.downloadImage')}
      variant="ghost"
      colorScheme="base"
      onPointerUp={onPointerUp}
    />
  );
});

ImageMenuItemDownload.displayName = 'ImageMenuItemDownload';
