import { IconMenuItem } from 'common/components/IconMenuItem';
import { useDeleteImageModalApi } from 'features/deleteImageModal/store/state';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

export const ImageMenuItemDelete = memo(() => {
  const { t } = useTranslation();
  const deleteImageModal = useDeleteImageModalApi();
  const imageDTO = useImageDTOContext();

  const onClick = useCallback(async () => {
    try {
      await deleteImageModal.delete([imageDTO.image_name]);
    } catch {
      // noop;
    }
  }, [deleteImageModal, imageDTO]);

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

ImageMenuItemDelete.displayName = 'ImageMenuItemDelete';
