import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useNewCanvasFromImage } from 'features/controlLayers/hooks/addLayerHooks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { toast } from 'features/toast/toast';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFileBold } from 'react-icons/pi';

export const ImageMenuItemNewCanvasFromImage = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const imageDTO = useImageDTOContext();
  const imageViewer = useImageViewer();
  const newCanvasFromImage = useNewCanvasFromImage();
  const isBusy = useCanvasIsBusy();

  const onClick = useCallback(() => {
    newCanvasFromImage(imageDTO);
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [dispatch, imageDTO, imageViewer, newCanvasFromImage, t]);

  return (
    <MenuItem icon={<PiFileBold />} onClickCapture={onClick} isDisabled={isBusy}>
      {t('controlLayers.newCanvasFromImage')}
    </MenuItem>
  );
});

ImageMenuItemNewCanvasFromImage.displayName = 'ImageMenuItemNewCanvasFromImage';
