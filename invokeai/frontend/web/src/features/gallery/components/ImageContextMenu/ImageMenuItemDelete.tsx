import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

export const ImageMenuItemDelete = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const imageDTO = useImageDTOContext();

  const onPointerUp = useCallback(() => {
    dispatch(imagesToDeleteSelected([imageDTO]));
  }, [dispatch, imageDTO]);

  return (
    <IconButton
      icon={<PiTrashSimpleBold />}
      onPointerUpCapture={onPointerUp}
      aria-label={t('gallery.deleteImage', { count: 1 })}
      tooltip={t('gallery.deleteImage', { count: 1 })}
      variant="ghost"
      colorScheme="red"
    />
  );
});

ImageMenuItemDelete.displayName = 'ImageMenuItemDelete';
