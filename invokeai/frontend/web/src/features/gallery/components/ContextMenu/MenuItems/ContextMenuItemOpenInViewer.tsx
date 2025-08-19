import { useAppDispatch } from 'app/store/storeHooks';
import { IconMenuItem } from 'common/components/IconMenuItem';
import { useItemDTOContext } from 'features/gallery/contexts/ItemDTOContext';
import { imageSelected, imageToCompareChanged } from 'features/gallery/store/gallerySlice';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { VIEWER_PANEL_ID } from 'features/ui/layouts/shared';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsOutBold } from 'react-icons/pi';
import { isImageDTO } from 'services/api/types';

export const ContextMenuItemOpenInViewer = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const itemDTO = useItemDTOContext();
  const onClick = useCallback(() => {
    if (isImageDTO(itemDTO)) {
    dispatch(imageToCompareChanged(null));
    dispatch(imageSelected(itemDTO.image_name));
    navigationApi.focusPanelInActiveTab(VIEWER_PANEL_ID);
    } else {
      // TODO: Implement video open in viewer
    }
  }, [dispatch, itemDTO]);

  return (
    <IconMenuItem
      icon={<PiArrowsOutBold />}
      onClickCapture={onClick}
      aria-label={t('common.openInViewer')}
      tooltip={t('common.openInViewer')}
    />
  );
});

ContextMenuItemOpenInViewer.displayName = 'ContextMenuItemOpenInViewer';
