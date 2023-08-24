import { skipToken } from '@reduxjs/toolkit/dist/query';
import { useCallback, useMemo, useState } from 'react';
import {
  useGetImageDTOQuery,
  useGetImageMetadataQuery,
} from '../../../services/api/endpoints/images';
import { useAppDispatch } from '../../../app/store/storeHooks';
import { setInitialCanvasImage } from '../../canvas/store/canvasSlice';
import { setActiveTab } from '../../ui/store/uiSlice';
import { useRecallParameters } from './useRecallParameters';
import { initialImageSelected } from '../store/actions';
import { useAppToaster } from '../../../app/components/Toaster';
import { t } from 'i18next';

type SelectedImage = {
  imageName: string;
  action: 'sendToImg2Img' | 'sendToCanvas' | 'useAllParameters';
};

export const usePreselectedImage = () => {
  const dispatch = useAppDispatch();
  const [imageNameForDto, setImageNameForDto] = useState<string | undefined>();
  const [imageNameForMetadata, setImageNameForMetadata] = useState<
    string | undefined
  >();
  const { recallAllParameters } = useRecallParameters();
  const toaster = useAppToaster();

  const { currentData: selectedImageDto, isError } = useGetImageDTOQuery(
    imageNameForDto ?? skipToken
  );

  const { currentData: selectedImageMetadata } = useGetImageMetadataQuery(
    imageNameForMetadata ?? skipToken
  );

  const handlePreselectedImage = useCallback(
    (selectedImage?: SelectedImage) => {
      if (!selectedImage) {
return;
}

      if (selectedImage.action === 'sendToCanvas') {
        setImageNameForDto(selectedImage?.imageName);
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
      }

      if (selectedImage.action === 'sendToImg2Img') {
        setImageNameForDto(selectedImage?.imageName);
        if (selectedImageDto) {
          dispatch(initialImageSelected(selectedImageDto));
        }
      }

      if (selectedImage.action === 'useAllParameters') {
        setImageNameForMetadata(selectedImage?.imageName);
        if (selectedImageMetadata) {
          recallAllParameters(selectedImageMetadata.metadata);
        }
      }
    },
    [
      dispatch,
      selectedImageDto,
      selectedImageMetadata,
      recallAllParameters,
      toaster,
    ]
  );

  return { handlePreselectedImage };
};
