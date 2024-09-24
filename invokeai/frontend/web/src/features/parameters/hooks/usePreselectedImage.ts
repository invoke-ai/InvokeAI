import { skipToken } from '@reduxjs/toolkit/query';
import { parseAndRecallAllMetadata } from 'features/metadata/util/handlers';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { useCallback, useEffect, useState } from 'react';
import { useGetImageDTOQuery, useGetImageMetadataQuery } from 'services/api/endpoints/images';
import { rasterLayerAdded } from '../../controlLayers/store/canvasSlice';
import { CanvasRasterLayerState } from '../../controlLayers/store/types';
import { imageDTOToImageObject } from '../../controlLayers/store/util';
import { sentImageToCanvas } from '../../gallery/store/actions';
import { setActiveTab } from '../../ui/store/uiSlice';
import { useAppDispatch, useAppSelector } from '../../../app/store/storeHooks';
import { selectBboxRect } from '../../controlLayers/store/selectors';
import { useImageViewer } from '../../gallery/components/ImageViewer/useImageViewer';
import { canvasReset } from '../../controlLayers/store/actions';
import { settingsSendToCanvasChanged } from '../../controlLayers/store/canvasSettingsSlice';

export const usePreselectedImage = (selectedImage?: {
  imageName: string;
  action: 'sendToImg2Img' | 'sendToCanvas' | 'useAllParameters';
}) => {
  const [isInit, setIsInit] = useState(false);
  const dispatch = useAppDispatch()
  const bboxRect = useAppSelector(selectBboxRect);
  const imageViewer = useImageViewer()
  const { currentData: selectedImageDto } = useGetImageDTOQuery(selectedImage?.imageName ?? skipToken);

  const { currentData: selectedImageMetadata } = useGetImageMetadataQuery(selectedImage?.imageName ?? skipToken);

  const handleSendToCanvas = useCallback(() => {
    if (isInit) {
      return
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
      imageViewer.close()
      toast({
        id: 'SENT_TO_CANVAS',
        title: t('toast.sentToUnifiedCanvas'),
        status: 'info',
      });
      setIsInit(true)

    }
  }, [selectedImageDto, dispatch, bboxRect, imageViewer, isInit]);

  const handleSendToImg2Img = useCallback(() => {
    if (isInit) {
      return
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
  }, [bboxRect.x, bboxRect.y, dispatch, selectedImageDto, imageViewer, t]);

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
