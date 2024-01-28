import { skipToken } from '@reduxjs/toolkit/query';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import type { TypesafeDraggableData, TypesafeDroppableData } from 'features/dnd/types';
import { clearInitialImage, selectGenerationSlice } from 'features/parameters/store/generationSlice';
import { memo, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

const selectInitialImage = createMemoizedSelector(selectGenerationSlice, (generation) => generation.initialImage);

const InitialImage = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const initialImage = useAppSelector(selectInitialImage);
  const isConnected = useAppSelector((s) => s.system.isConnected);

  const { currentData: imageDTO, isError } = useGetImageDTOQuery(initialImage?.imageName ?? skipToken);

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
      dropLabel={t('toast.setInitialImage')}
      noContentFallback={<IAINoContentFallback label={t('parameters.invoke.noInitialImageSelected')} />}
      dataTestId="initial-image"
    />
  );
};

export default memo(InitialImage);
