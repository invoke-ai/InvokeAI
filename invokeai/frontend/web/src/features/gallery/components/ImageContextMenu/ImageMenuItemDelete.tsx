import { MenuItem } from '@invoke-ai/ui-library';
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

  const onClick = useCallback(() => {
    dispatch(imagesToDeleteSelected([imageDTO]));
  }, [dispatch, imageDTO]);

  return (
    <MenuItem isDestructive icon={<PiTrashSimpleBold />} onClickCapture={onClick}>
      {t('gallery.deleteImage', { count: 1 })}
    </MenuItem>
  );
});

ImageMenuItemDelete.displayName = 'ImageMenuItemDelete';
