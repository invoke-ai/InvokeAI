import { IconMenuItem } from 'common/components/IconMenuItem';
import { useDownloadItem } from 'common/hooks/useDownloadImage';
import { useVideoDTOContext } from 'features/gallery/contexts/VideoDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadSimpleBold } from 'react-icons/pi';

export const ContextMenuItemDownloadVideo = memo(() => {
  const { t } = useTranslation();
  const videoDTO = useVideoDTOContext();
  const { downloadItem } = useDownloadItem();

  const onClick = useCallback(() => {
    downloadItem(videoDTO.video_url, videoDTO.video_name);
  }, [downloadItem, videoDTO]);

  return (
    <IconMenuItem
      icon={<PiDownloadSimpleBold />}
      aria-label={t('gallery.download')}
      tooltip={t('gallery.download')}
      onClick={onClick}
    />
  );
});

ContextMenuItemDownloadVideo.displayName = 'ContextMenuItemDownloadVideo';
