import { useAppDispatch } from 'app/store/storeHooks';
import { IconMenuItem } from 'common/components/IconMenuItem';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

export const ImageMenuItemDelete = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const imageDTO = useImageDTOContext();

  const onClick = useCallback(() => {
    dispatch(imagesToDeleteSelected([imageDTO]));
  }, [dispatch, imageDTO]);

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
