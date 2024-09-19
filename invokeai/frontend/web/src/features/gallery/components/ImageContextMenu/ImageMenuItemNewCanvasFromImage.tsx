import { MenuItem } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { canvasReset } from 'features/controlLayers/store/actions';
import { rasterLayerAdded } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { CanvasRasterLayerState } from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { toast } from 'features/toast/toast';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFileBold } from 'react-icons/pi';

const selectBboxRect = createSelector(selectCanvasSlice, (canvas) => canvas.bbox.rect);

export const ImageMenuItemNewCanvasFromImage = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const imageDTO = useImageDTOContext();
  const bboxRect = useAppSelector(selectBboxRect);
  const imageViewer = useImageViewer();

  const handleSendToCanvas = useCallback(() => {
    const imageObject = imageDTOToImageObject(imageDTO);
    const overrides: Partial<CanvasRasterLayerState> = {
      position: { x: bboxRect.x, y: bboxRect.y },
      objects: [imageObject],
    };
    dispatch(canvasReset());
    dispatch(rasterLayerAdded({ overrides, isSelected: true }));
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [bboxRect.x, bboxRect.y, dispatch, imageDTO, imageViewer, t]);

  return (
    <MenuItem icon={<PiFileBold />} onClickCapture={handleSendToCanvas}>
      {t('controlLayers.newCanvasFromImage')}
    </MenuItem>
  );
});

ImageMenuItemNewCanvasFromImage.displayName = 'ImageMenuItemNewCanvasFromImage';
