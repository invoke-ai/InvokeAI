import { skipToken } from '@reduxjs/toolkit/query';
import { parseAndRecallAllMetadata } from 'features/metadata/util/handlers';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { useCallback, useEffect } from 'react';
import { useGetImageDTOQuery, useGetImageMetadataQuery } from 'services/api/endpoints/images';

export const usePreselectedImage = (selectedImage?: {
  imageName: string;
  action: 'sendToImg2Img' | 'sendToCanvas' | 'useAllParameters';
}) => {
  const { currentData: selectedImageDto } = useGetImageDTOQuery(selectedImage?.imageName ?? skipToken);

  const { currentData: selectedImageMetadata } = useGetImageMetadataQuery(selectedImage?.imageName ?? skipToken);

  const handleSendToCanvas = useCallback(() => {
    if (selectedImageDto) {
      // TODO(psyche): handle send to canvas
      // dispatch(setInitialCanvasImage(selectedImageDto, optimalDimension));
      // dispatch(setActiveTab('canvas'));
      toast({
        id: 'SENT_TO_CANVAS',
        title: t('toast.sentToUnifiedCanvas'),
        status: 'info',
      });
    }
  }, [selectedImageDto]);

  const handleSendToImg2Img = useCallback(() => {
    if (selectedImageDto) {
      // TODO(psyche): handle send to img2img
      // dispatch(iiLayerAdded(selectedImageDto));
      // dispatch(setActiveTab('generation'));
    }
  }, [selectedImageDto]);

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
