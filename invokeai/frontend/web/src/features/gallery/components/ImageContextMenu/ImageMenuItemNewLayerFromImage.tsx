import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { NewLayerIcon } from 'features/controlLayers/components/common/icons';
import { useNewRasterLayerFromImage } from 'features/controlLayers/hooks/addLayerHooks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { sentImageToCanvas } from 'features/gallery/store/actions';
import { toast } from 'features/toast/toast';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const ImageMenuItemNewLayerFromImage = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const imageDTO = useImageDTOContext();
  const imageViewer = useImageViewer();
  const newRasterLayerFromImage = useNewRasterLayerFromImage();
  const isBusy = useCanvasIsBusy();

  const onClick = useCallback(() => {
    dispatch(sentImageToCanvas());
    newRasterLayerFromImage(imageDTO);
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [dispatch, imageDTO, imageViewer, newRasterLayerFromImage, t]);

  return (
    <MenuItem icon={<NewLayerIcon />} onClickCapture={onClick} isDisabled={isBusy}>
      {t('controlLayers.newLayerFromImage')}
    </MenuItem>
  );
});

ImageMenuItemNewLayerFromImage.displayName = 'ImageMenuItemNewLayerFromImage';
