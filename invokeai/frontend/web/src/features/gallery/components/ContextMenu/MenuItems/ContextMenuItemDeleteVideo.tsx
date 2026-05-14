import { IconMenuItem } from 'common/components/IconMenuItem';
import { useDeleteVideoModalApi } from 'features/deleteVideoModal/store/state';
import { useVideoDTOContext } from 'features/gallery/contexts/VideoDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

export const ContextMenuItemDeleteVideo = memo(() => {
  const { t } = useTranslation();
  const videoDTO = useVideoDTOContext();
  const deleteVideoModal = useDeleteVideoModalApi();

  const onClick = useCallback(async () => {
    try {
      await deleteVideoModal.delete([videoDTO.video_name]);
    } catch {
      // noop — user canceled the confirm dialog.
    }
  }, [deleteVideoModal, videoDTO.video_name]);

  return (
    <IconMenuItem
      icon={<PiTrashSimpleBold />}
      onClickCapture={onClick}
      aria-label={t('gallery.deleteVideo', { count: 1 })}
      tooltip={t('gallery.deleteVideo', { count: 1 })}
      isDestructive
    />
  );
});

ContextMenuItemDeleteVideo.displayName = 'ContextMenuItemDeleteVideo';
