import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { boardIdSelected } from 'features/gallery/store/gallerySlice';
import { IMAGE_CATEGORIES } from 'features/gallery/store/types';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { useGalleryPanel } from 'features/ui/layouts/use-gallery-panel';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo, useCallback, useMemo } from 'react';
import { flushSync } from 'react-dom';
import { useTranslation } from 'react-i18next';
import { PiCrosshairBold } from 'react-icons/pi';

export const ContextMenuItemLocateInGalery = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const imageDTO = useImageDTOContext();
  const activeTab = useAppSelector(selectActiveTab);
  const galleryPanel = useGalleryPanel(activeTab);

  const isGalleryImage = useMemo(() => {
    return !imageDTO.is_intermediate;
  }, [imageDTO]);

  const onClick = useCallback(() => {
    navigationApi.expandRightPanel();
    galleryPanel.expand();
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
  }, [dispatch, galleryPanel, imageDTO]);

  return (
    <MenuItem icon={<PiCrosshairBold />} onClickCapture={onClick} isDisabled={!isGalleryImage}>
      {t('boards.locateInGalery')}
    </MenuItem>
  );
});

ContextMenuItemLocateInGalery.displayName = 'ContextMenuItemLocateInGalery';
