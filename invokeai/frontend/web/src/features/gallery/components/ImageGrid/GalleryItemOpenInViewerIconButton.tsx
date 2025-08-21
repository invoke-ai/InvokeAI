import { useAppDispatch } from 'app/store/storeHooks';
import { DndImageIcon } from 'features/dnd/DndImageIcon';
import { imageSelected, imageToCompareChanged } from 'features/gallery/store/gallerySlice';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { VIEWER_PANEL_ID } from 'features/ui/layouts/shared';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsOutBold } from 'react-icons/pi';
import { type ImageDTO, isImageDTO, type VideoDTO } from 'services/api/types';

type Props = {
  itemDTO: ImageDTO | VideoDTO;
};

export const GalleryItemOpenInViewerIconButton = memo(({ itemDTO }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const onClick = useCallback(() => {
    if (isImageDTO(itemDTO)) {
      dispatch(imageToCompareChanged(null));
      dispatch(imageSelected(itemDTO.image_name));
    } else {
      // dispatch(videoToCompareChanged(null));
      // dispatch(videoSelected(itemDTO.video_id));
    }
    navigationApi.focusPanelInActiveTab(VIEWER_PANEL_ID);
  }, [dispatch, itemDTO]);

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
