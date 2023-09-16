import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/dist/query';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIDndImage from 'common/components/IAIDndImage';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import {
  TypesafeDraggableData,
  TypesafeDroppableData,
} from 'features/dnd/types';
import { memo, useMemo } from 'react';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

const selector = createSelector(
  [stateSelector],
  (state) => {
    const { initialImage } = state.generation;
    return {
      initialImage,
      isResetButtonDisabled: !initialImage,
    };
  },
  defaultSelectorOptions
);

const InitialImage = () => {
  const { initialImage } = useAppSelector(selector);

  const { currentData: imageDTO } = useGetImageDTOQuery(
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

export default memo(InitialImage);
