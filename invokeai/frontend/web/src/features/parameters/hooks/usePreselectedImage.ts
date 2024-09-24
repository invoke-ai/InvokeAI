import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { canvasReset } from 'features/controlLayers/store/actions';
import { settingsSendToCanvasChanged } from 'features/controlLayers/store/canvasSettingsSlice';
import { rasterLayerAdded } from 'features/controlLayers/store/canvasSlice';
import { selectBboxRect } from 'features/controlLayers/store/selectors';
import type { CanvasRasterLayerState } from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { sentImageToCanvas } from 'features/gallery/store/actions';
import { parseAndRecallAllMetadata } from 'features/metadata/util/handlers';
import { toast } from 'features/toast/toast';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetImageDTOQuery, useGetImageMetadataQuery } from 'services/api/endpoints/images';

export const usePreselectedImage = (selectedImage?: {
  imageName: string;
  action: 'sendToImg2Img' | 'sendToCanvas' | 'useAllParameters';
}) => {
  const [isInit, setIsInit] = useState(false);
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const bboxRect = useAppSelector(selectBboxRect);
  const imageViewer = useImageViewer();
  const { currentData: selectedImageDto } = useGetImageDTOQuery(selectedImage?.imageName ?? skipToken);

  const { currentData: selectedImageMetadata } = useGetImageMetadataQuery(selectedImage?.imageName ?? skipToken);

  const handleSendToCanvas = useCallback(() => {
    if (isInit) {
      return;
    }
    if (selectedImageDto) {
      const imageObject = imageDTOToImageObject(selectedImageDto);
      const overrides: Partial<CanvasRasterLayerState> = {
        position: { x: bboxRect.x, y: bboxRect.y },
        objects: [imageObject],
      };
      dispatch(sentImageToCanvas());
      dispatch(rasterLayerAdded({ overrides, isSelected: true }));
      dispatch(setActiveTab('canvas'));
      dispatch(settingsSendToCanvasChanged(true));
      imageViewer.close();
      toast({
        id: 'SENT_TO_CANVAS',
        title: t('toast.sentToUnifiedCanvas'),
        status: 'info',
      });
      setIsInit(true);
    }
  }, [selectedImageDto, dispatch, bboxRect, imageViewer, isInit, t]);

  const handleSendToImg2Img = useCallback(() => {
    if (isInit) {
      return;
    }
    if (selectedImageDto) {
      const imageObject = imageDTOToImageObject(selectedImageDto);
      const overrides: Partial<CanvasRasterLayerState> = {
        position: { x: bboxRect.x, y: bboxRect.y },
        objects: [imageObject],
      };
      dispatch(canvasReset());
      dispatch(rasterLayerAdded({ overrides, isSelected: true }));
      dispatch(setActiveTab('canvas'));
      dispatch(settingsSendToCanvasChanged(false));
      imageViewer.close();
      toast({
        id: 'SENT_TO_CANVAS',
        title: t('toast.sentToCanvas'),
        status: 'success',
      });
    }
  }, [bboxRect.x, bboxRect.y, dispatch, selectedImageDto, imageViewer, t, isInit]);

  const handleUseAllMetadata = useCallback(() => {
    if (selectedImageMetadata) {
      parseAndRecallAllMetadata(selectedImageMetadata, true);
    }
  }, [selectedImageMetadata]);

  useEffect(() => {
    if (selectedImage && selectedImage.action === 'sendToCanvas') {
      handleSendToCanvas();
    }
  }, [selectedImage, handleSendToCanvas]);

  useEffect(() => {
    if (selectedImage && selectedImage.action === 'sendToImg2Img') {
      handleSendToImg2Img();
    }
  }, [selectedImage, handleSendToImg2Img]);

  useEffect(() => {
    if (selectedImage && selectedImage.action === 'useAllParameters') {
      handleUseAllMetadata();
    }
  }, [selectedImage, handleUseAllMetadata]);

  return { handleSendToCanvas, handleSendToImg2Img, handleUseAllMetadata };
};
