import { IconMenuItem } from 'common/components/IconMenuItem';
import { useDownloadItem } from 'common/hooks/useDownloadImage';
import { useItemDTOContext } from 'features/gallery/contexts/ItemDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadSimpleBold } from 'react-icons/pi';
import { isImageDTO } from 'services/api/types';

export const ContextMenuItemDownload = memo(() => {
  const { t } = useTranslation();
  const itemDTO = useItemDTOContext();
  const { downloadItem } = useDownloadItem();

  const onClick = useCallback(() => {
    if (isImageDTO(itemDTO)) {
      downloadItem(itemDTO.image_url, itemDTO.image_name);
    } else {
      downloadItem(itemDTO.video_url, itemDTO.video_id);
    }
  }, [downloadItem, itemDTO]);

  return (
    <IconMenuItem
      icon={<PiDownloadSimpleBold />}
      aria-label={t('gallery.download')}
      tooltip={t('gallery.download')}
      onClick={onClick}
    />
  );
});

ContextMenuItemDownload.displayName = 'ContextMenuItemDownload';
