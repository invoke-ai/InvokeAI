import { skipToken } from '@reduxjs/toolkit/dist/query';
import { t } from 'i18next';
import { useCallback, useEffect } from 'react';
import { useAppToaster } from '../../../app/components/Toaster';
import { useAppDispatch } from '../../../app/store/storeHooks';
import {
  useGetImageDTOQuery,
  useGetImageMetadataQuery,
} from '../../../services/api/endpoints/images';
import { setInitialCanvasImage } from '../../canvas/store/canvasSlice';
import { setActiveTab } from '../../ui/store/uiSlice';
import { initialImageSelected } from '../store/actions';
import { useRecallParameters } from './useRecallParameters';

export const usePreselectedImage = (selectedImage?: {
  imageName: string;
  action: 'sendToImg2Img' | 'sendToCanvas' | 'useAllParameters';
}) => {
  const dispatch = useAppDispatch();

  const { recallAllParameters } = useRecallParameters();
  const toaster = useAppToaster();

  const { currentData: selectedImageDto } = useGetImageDTOQuery(
    selectedImage?.imageName ?? skipToken
  );

  const { currentData: selectedImageMetadata } = useGetImageMetadataQuery(
    selectedImage?.imageName ?? skipToken
  );

  const handleSendToCanvas = useCallback(() => {
    if (selectedImageDto) {
      dispatch(setInitialCanvasImage(selectedImageDto));
      dispatch(setActiveTab('unifiedCanvas'));
      toaster({
        title: t('toast.sentToUnifiedCanvas'),
        status: 'info',
        duration: 2500,
        isClosable: true,
      });
    }
  }, [dispatch, toaster, selectedImageDto]);

  const handleSendToImg2Img = useCallback(() => {
    if (selectedImageDto) {
      dispatch(initialImageSelected(selectedImageDto));
    }
  }, [dispatch, selectedImageDto]);

  const handleUseAllMetadata = useCallback(() => {
    if (selectedImageMetadata) {
      recallAllParameters(selectedImageMetadata);
    }
    // disabled because `recallAllParameters` changes the model, but its dep to prepare LoRAs has model as a dep. this introduces circular logic that causes infinite re-renders
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
