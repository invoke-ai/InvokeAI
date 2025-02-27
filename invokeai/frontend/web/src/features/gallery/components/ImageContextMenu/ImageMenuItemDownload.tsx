import { IconMenuItem } from 'common/components/IconMenuItem';
import { useDownloadImage } from 'common/hooks/useDownloadImage';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadSimpleBold } from 'react-icons/pi';

export const ImageMenuItemDownload = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useImageDTOContext();
  const { downloadImage } = useDownloadImage();

  const onClick = useCallback(() => {
    downloadImage(imageDTO.image_url, imageDTO.image_name);
  }, [downloadImage, imageDTO.image_name, imageDTO.image_url]);

  return (
    <IconMenuItem
      icon={<PiDownloadSimpleBold />}
      aria-label={t('parameters.downloadImage')}
      tooltip={t('parameters.downloadImage')}
      onClick={onClick}
    />
  );
});

ImageMenuItemDownload.displayName = 'ImageMenuItemDownload';
