import { IconMenuItem } from 'common/components/IconMenuItem';
import { useVideoDTOContext } from 'features/gallery/contexts/VideoDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';
import { useDeleteVideoMutation } from 'services/api/endpoints/videos';

export const ContextMenuItemDeleteVideo = memo(() => {
  const { t } = useTranslation();
  const videoDTO = useVideoDTOContext();
  const [deleteVideo] = useDeleteVideoMutation();

  const onClick = useCallback(() => {
    // Confirm-then-delete via the native dialog. Videos can't be referenced from canvas/nodes/
    // refs the way images can, so the image modal's usage analysis is unnecessary; a one-step
    // confirm matches "minimal" scope.
    if (window.confirm(t('gallery.deleteVideoConfirmation', { defaultValue: 'Delete this video?' }))) {
      deleteVideo({ video_name: videoDTO.video_name });
    }
  }, [deleteVideo, t, videoDTO.video_name]);

  return (
    <IconMenuItem
      icon={<PiTrashSimpleBold />}
      onClickCapture={onClick}
      aria-label={t('gallery.deleteImage', { count: 1 })}
      tooltip={t('gallery.deleteImage', { count: 1 })}
      isDestructive
    />
  );
});

ContextMenuItemDeleteVideo.displayName = 'ContextMenuItemDeleteVideo';
