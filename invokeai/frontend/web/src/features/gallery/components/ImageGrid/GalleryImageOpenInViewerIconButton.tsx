import { useAppDispatch } from 'app/store/storeHooks';
import { DndImageIcon } from 'features/dnd/DndImageIcon';
import { imageSelected, imageToCompareChanged } from 'features/gallery/store/gallerySlice';
import { useAutoLayoutContext } from 'features/ui/layouts/auto-layout-context';
import { VIEWER_PANEL_ID } from 'features/ui/layouts/shared';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsOutBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

type Props = {
  imageDTO: ImageDTO;
};

export const GalleryImageOpenInViewerIconButton = memo(({ imageDTO }: Props) => {
  const dispatch = useAppDispatch();
  const { focusPanel } = useAutoLayoutContext();
  const { t } = useTranslation();

  const onClick = useCallback(() => {
    dispatch(imageToCompareChanged(null));
    dispatch(imageSelected(imageDTO));
    focusPanel(VIEWER_PANEL_ID);
  }, [dispatch, focusPanel, imageDTO]);

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

GalleryImageOpenInViewerIconButton.displayName = 'GalleryImageOpenInViewerIconButton';
