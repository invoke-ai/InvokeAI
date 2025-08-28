import { IconMenuItem } from 'common/components/IconMenuItem';
import { useDownloadImage } from 'common/hooks/useDownloadImage';
import { useItemDTOContext } from 'features/gallery/contexts/ItemDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadSimpleBold } from 'react-icons/pi';
import { isImageDTO } from 'services/api/types';

export const ContextMenuItemDownload = memo(() => {
  const { t } = useTranslation();
  const itemDTO = useItemDTOContext();
  const { downloadImage } = useDownloadImage();

  const onClick = useCallback(() => {
    if (isImageDTO(itemDTO)) {
      downloadImage(itemDTO.image_url, itemDTO.image_name);
    } else {
      // downloadVideo(itemDTO.video_url, itemDTO.video_id);
    }
  }, [downloadImage, itemDTO]);

  return (
    <IconMenuItem
      icon={<PiDownloadSimpleBold />}
      aria-label={t('parameters.downloadImage')}
      tooltip={t('parameters.downloadImage')}
      onClick={onClick}
    />
  );
});

ContextMenuItemDownload.displayName = 'ContextMenuItemDownload';
