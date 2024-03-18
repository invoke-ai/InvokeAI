import { skipToken } from '@reduxjs/toolkit/query';
import { useAppToaster } from 'app/components/Toaster';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import { parseAndRecallAllMetadata } from 'features/metadata/util/handlers';
import { initialImageSelected } from 'features/parameters/store/actions';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { t } from 'i18next';
import { useCallback, useEffect } from 'react';
import { useGetImageDTOQuery, useGetImageMetadataQuery } from 'services/api/endpoints/images';

export const usePreselectedImage = (selectedImage?: {
  imageName: string;
  action: 'sendToImg2Img' | 'sendToCanvas' | 'useAllParameters';
}) => {
  const dispatch = useAppDispatch();
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const toaster = useAppToaster();

  const { currentData: selectedImageDto } = useGetImageDTOQuery(selectedImage?.imageName ?? skipToken);

  const { currentData: selectedImageMetadata } = useGetImageMetadataQuery(selectedImage?.imageName ?? skipToken);

  const handleSendToCanvas = useCallback(() => {
    if (selectedImageDto) {
      dispatch(setInitialCanvasImage(selectedImageDto, optimalDimension));
      dispatch(setActiveTab('unifiedCanvas'));
      toaster({
        title: t('toast.sentToUnifiedCanvas'),
        status: 'info',
        duration: 2500,
        isClosable: true,
      });
    }
  }, [selectedImageDto, dispatch, optimalDimension, toaster]);

  const handleSendToImg2Img = useCallback(() => {
    if (selectedImageDto) {
      dispatch(initialImageSelected(selectedImageDto));
    }
  }, [dispatch, selectedImageDto]);

  const handleUseAllMetadata = useCallback(() => {
    if (selectedImageMetadata) {
      parseAndRecallAllMetadata(selectedImageMetadata);
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
