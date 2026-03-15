import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { boardIdSelected } from 'features/gallery/store/gallerySlice';
import { IMAGE_CATEGORIES } from 'features/gallery/store/types';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { memo, useCallback, useMemo } from 'react';
import { flushSync } from 'react-dom';
import { useTranslation } from 'react-i18next';
import { PiCrosshairBold } from 'react-icons/pi';

export const ContextMenuItemLocateInGalery = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const imageDTO = useImageDTOContext();

  const isGalleryImage = useMemo(() => {
    return !imageDTO.is_intermediate;
  }, [imageDTO]);

  const onClick = useCallback(() => {
    navigationApi.expandBottomPanel();
    flushSync(() => {
      dispatch(
        boardIdSelected({
          boardId: imageDTO.board_id ?? 'none',
          select: {
            selection: [imageDTO.image_name],
            galleryView: IMAGE_CATEGORIES.includes(imageDTO.image_category) ? 'images' : 'assets',
          },
        })
      );
    });
  }, [dispatch, imageDTO]);

  return (
    <MenuItem icon={<PiCrosshairBold />} onClickCapture={onClick} isDisabled={!isGalleryImage}>
      {t('boards.locateInGalery')}
    </MenuItem>
  );
});

ContextMenuItemLocateInGalery.displayName = 'ContextMenuItemLocateInGalery';
