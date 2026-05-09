import { useAppDispatch } from 'app/store/storeHooks';
import { DndImageIcon } from 'features/dnd/DndImageIcon';
import { imageSelected, imageToCompareChanged } from 'features/gallery/store/gallerySlice';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { VIEWER_PANEL_ID } from 'features/ui/layouts/shared';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsOutBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

type Props = {
  imageDTO: ImageDTO;
};

export const GalleryItemOpenInViewerIconButton = memo(({ imageDTO }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const onClick = useCallback(() => {
    dispatch(imageToCompareChanged(null));
    dispatch(imageSelected(imageDTO.image_name));
    navigationApi.focusPanelInActiveTab(VIEWER_PANEL_ID);
  }, [dispatch, imageDTO]);

  return (
    <DndImageIcon
      onClick={onClick}
      icon={<PiArrowsOutBold />}
      tooltip={t('gallery.openInViewer')}
      position="absolute"
      insetBlockStart={2}
      insetInlineStart={2}
    />
  );
});

GalleryItemOpenInViewerIconButton.displayName = 'GalleryItemOpenInViewerIconButton';
