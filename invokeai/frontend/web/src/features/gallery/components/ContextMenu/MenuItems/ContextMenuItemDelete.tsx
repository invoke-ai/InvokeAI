import { IconMenuItem } from 'common/components/IconMenuItem';
import { useDeleteImageModalApi } from 'features/deleteImageModal/store/state';
import { useItemDTOContext } from 'features/gallery/contexts/ItemDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';
import { isImageDTO } from 'services/api/types';

export const ContextMenuItemDelete = memo(() => {
  const { t } = useTranslation();
  const deleteImageModal = useDeleteImageModalApi();
  const itemDTO = useItemDTOContext();

  const onClick = useCallback(async () => {
    try {
      if (isImageDTO(itemDTO)) {
        await deleteImageModal.delete([itemDTO.image_name]);
      } else {
        // await deleteVideoModal.delete([itemDTO.video_id]);
      }
    } catch {
      // noop;
    }
  }, [deleteImageModal, itemDTO]);

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

ContextMenuItemDelete.displayName = 'ContextMenuItemDelete';
