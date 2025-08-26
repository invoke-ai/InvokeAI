import { IconMenuItem } from 'common/components/IconMenuItem';
import { useDeleteVideoModalApi } from 'features/deleteVideoModal/store/state';
import { useItemDTOContext } from 'features/gallery/contexts/ItemDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';
import { isVideoDTO } from 'services/api/types';

export const ContextMenuItemDeleteVideo = memo(() => {
  const { t } = useTranslation();
  const deleteVideoModal = useDeleteVideoModalApi();
  const itemDTO = useItemDTOContext();

  const onClick = useCallback(async () => {
    try {
      if (isVideoDTO(itemDTO)) {
        await deleteVideoModal.delete([itemDTO.video_id]);
      }
    } catch {
      // noop;
    }
  }, [deleteVideoModal, itemDTO]);

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
