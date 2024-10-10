import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useNewImg2ImgCanvasFromImage } from 'features/controlLayers/hooks/addLayerHooks';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { toast } from 'features/toast/toast';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFileImageBold } from 'react-icons/pi';

export const ImageMenuItemsNewImg2ImgCanvasFromImage = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const imageDTO = useImageDTOContext();
  const imageViewer = useImageViewer();
  const newImg2ImgCanvasFromImage = useNewImg2ImgCanvasFromImage();

  const onClick = useCallback(() => {
    newImg2ImgCanvasFromImage(imageDTO);
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [dispatch, imageDTO, imageViewer, newImg2ImgCanvasFromImage, t]);

  return (
    <MenuItem icon={<PiFileImageBold />} onClickCapture={onClick}>
      {t('controlLayers.newImg2ImgCanvasFromImage')}
    </MenuItem>
  );
});

ImageMenuItemsNewImg2ImgCanvasFromImage.displayName = 'ImageMenuItemsNewImg2ImgCanvasFromImage';
