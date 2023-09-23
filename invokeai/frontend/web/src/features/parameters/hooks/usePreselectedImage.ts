import { skipToken } from '@reduxjs/toolkit/dist/query';
import { CoreMetadata } from 'features/nodes/types/types';
import { t } from 'i18next';
import { useCallback } from 'react';
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

export const usePreselectedImage = (imageName?: string) => {
  const dispatch = useAppDispatch();

  const { recallAllParameters } = useRecallParameters();
  const toaster = useAppToaster();

  const { currentData: selectedImageDto } = useGetImageDTOQuery(
    imageName ?? skipToken
  );

  const { currentData: selectedImageMetadata } = useGetImageMetadataQuery(
    imageName ?? skipToken
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
      recallAllParameters(selectedImageMetadata.metadata as CoreMetadata);
    }
    // disabled because `recallAllParameters` changes the model, but its dep to prepare LoRAs has model as a dep. this introduces circular logic that causes infinite re-renders
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedImageMetadata]);

  return { handleSendToCanvas, handleSendToImg2Img, handleUseAllMetadata };
};
