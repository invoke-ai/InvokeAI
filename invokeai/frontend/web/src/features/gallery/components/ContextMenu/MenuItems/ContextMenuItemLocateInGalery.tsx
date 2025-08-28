import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useItemDTOContext } from 'features/gallery/contexts/ItemDTOContext';
import { boardIdSelected } from 'features/gallery/store/gallerySlice';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { useGalleryPanel } from 'features/ui/layouts/use-gallery-panel';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo, useCallback, useMemo } from 'react';
import { flushSync } from 'react-dom';
import { useTranslation } from 'react-i18next';
import { PiCrosshairBold } from 'react-icons/pi';
import { isImageDTO } from 'services/api/types';

export const ContextMenuItemLocateInGalery = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const itemDTO = useItemDTOContext();
  const activeTab = useAppSelector(selectActiveTab);
  const galleryPanel = useGalleryPanel(activeTab);

  const isGalleryImage = useMemo(() => {
    return !itemDTO.is_intermediate;
  }, [itemDTO]);

  const onClick = useCallback(() => {
    if (isImageDTO(itemDTO)) {
      navigationApi.expandRightPanel();
      galleryPanel.expand();
      flushSync(() => {
        dispatch(boardIdSelected({ boardId: itemDTO.board_id ?? 'none', selectedImageName: itemDTO.image_name }));
      });
    } else {
      // TODO: Implement video locate in gallery
    }
  }, [dispatch, galleryPanel, itemDTO]);

  return (
    <MenuItem icon={<PiCrosshairBold />} onClickCapture={onClick} isDisabled={!isGalleryImage}>
      {t('boards.locateInGalery')}
    </MenuItem>
  );
});

ContextMenuItemLocateInGalery.displayName = 'ContextMenuItemLocateInGalery';
