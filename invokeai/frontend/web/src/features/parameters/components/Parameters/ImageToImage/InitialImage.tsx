import { skipToken } from '@reduxjs/toolkit/query';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import {
  TypesafeDraggableData,
  TypesafeDroppableData,
} from 'features/dnd/types';
import { clearInitialImage } from 'features/parameters/store/generationSlice';
import { memo, useEffect, useMemo } from 'react';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

const selector = createMemoizedSelector([stateSelector], (state) => {
  const { initialImage } = state.generation;
  const { isConnected } = state.system;

  return {
    initialImage,
    isResetButtonDisabled: !initialImage,
    isConnected,
  };
});

const InitialImage = () => {
  const dispatch = useAppDispatch();
  const { initialImage, isConnected } = useAppSelector(selector);

  const { currentData: imageDTO, isError } = useGetImageDTOQuery(
    initialImage?.imageName ?? skipToken
  );

  const draggableData = useMemo<TypesafeDraggableData | undefined>(() => {
    if (imageDTO) {
      return {
        id: 'initial-image',
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO },
      };
    }
  }, [imageDTO]);

  const droppableData = useMemo<TypesafeDroppableData | undefined>(
    () => ({
      id: 'initial-image',
      actionType: 'SET_INITIAL_IMAGE',
    }),
    []
  );

  useEffect(() => {
    if (isError && isConnected) {
      // The image doesn't exist, reset init image
      dispatch(clearInitialImage());
    }
  }, [dispatch, isConnected, isError]);

  return (
    <IAIDndImage
      imageDTO={imageDTO}
      droppableData={droppableData}
      draggableData={draggableData}
      isUploadDisabled={true}
      fitContainer
      dropLabel="Set as Initial Image"
      noContentFallback={
        <IAINoContentFallback label="No initial image selected" />
      }
      dataTestId="initial-image"
    />
  );
};

export default memo(InitialImage);
