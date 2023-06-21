import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  clearInitialImage,
  initialImageChanged,
} from 'features/parameters/store/generationSlice';
import { useCallback } from 'react';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIDndImage from 'common/components/IAIDndImage';
import { ImageDTO } from 'services/api';
import { IAIImageFallback } from 'common/components/IAIImageFallback';
import { useGetImageDTOQuery } from 'services/apiSlice';
import { skipToken } from '@reduxjs/toolkit/dist/query';

const selector = createSelector(
  [generationSelector],
  (generation) => {
    const { initialImage } = generation;
    return {
      initialImage,
    };
  },
  defaultSelectorOptions
);

const InitialImagePreview = () => {
  const { initialImage } = useAppSelector(selector);
  const dispatch = useAppDispatch();

  const {
    data: image,
    isLoading,
    isError,
    isSuccess,
  } = useGetImageDTOQuery(initialImage ?? skipToken);

  const handleDrop = useCallback(
    ({ image_name }: ImageDTO) => {
      if (image_name === initialImage) {
        return;
      }
      dispatch(initialImageChanged(image_name));
    },
    [dispatch, initialImage]
  );

  const handleReset = useCallback(() => {
    dispatch(clearInitialImage());
  }, [dispatch]);

  return (
    <Flex
      sx={{
        width: 'full',
        height: 'full',
        position: 'absolute',
        alignItems: 'center',
        justifyContent: 'center',
        p: 4,
      }}
    >
      <IAIDndImage
        image={image}
        onDrop={handleDrop}
        onReset={handleReset}
        fallback={<IAIImageFallback sx={{ bg: 'none' }} />}
        postUploadAction={{ type: 'SET_INITIAL_IMAGE' }}
        withResetIcon
      />
    </Flex>
  );
};

export default InitialImagePreview;
