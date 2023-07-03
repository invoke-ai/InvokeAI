import { Flex, Icon, Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useMemo } from 'react';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIDndImage from 'common/components/IAIDndImage';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { skipToken } from '@reduxjs/toolkit/dist/query';
import { FaImage } from 'react-icons/fa';
import { stateSelector } from 'app/store/store';
import {
  TypesafeDraggableData,
  TypesafeDroppableData,
} from 'app/components/ImageDnd/typesafeDnd';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';

const selector = createSelector(
  [stateSelector],
  (state) => {
    const { initialImage } = state.generation;
    const { asInitialImage: useBatchAsInitialImage, imageNames } = state.batch;
    return {
      initialImage,
      useBatchAsInitialImage,
      isResetButtonDisabled: useBatchAsInitialImage
        ? imageNames.length === 0
        : !initialImage,
    };
  },
  defaultSelectorOptions
);

const InitialImage = () => {
  const { initialImage } = useAppSelector(selector);

  const {
    currentData: imageDTO,
    isLoading,
    isError,
    isSuccess,
  } = useGetImageDTOQuery(initialImage?.imageName ?? skipToken);

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
    />
  );
};

export default InitialImage;
