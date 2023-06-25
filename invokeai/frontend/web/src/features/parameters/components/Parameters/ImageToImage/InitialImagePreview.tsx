import { Flex, Spacer, Text } from '@chakra-ui/react';
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
import { ImageDTO } from 'services/api/types';
import { IAIImageLoadingFallback } from 'common/components/IAIImageFallback';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { skipToken } from '@reduxjs/toolkit/dist/query';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaUndo, FaUpload } from 'react-icons/fa';
import useImageUploader from 'common/hooks/useImageUploader';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';

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
  const { openUploader } = useImageUploader();

  const {
    currentData: image,
    isLoading,
    isError,
    isSuccess,
  } = useGetImageDTOQuery(initialImage?.imageName ?? skipToken);

  const { getUploadButtonProps, getUploadInputProps } = useImageUploadButton({
    postUploadAction: { type: 'SET_INITIAL_IMAGE' },
  });

  const handleDrop = useCallback(
    (droppedImage: ImageDTO) => {
      if (droppedImage.image_name === initialImage?.imageName) {
        return;
      }
      dispatch(initialImageChanged(droppedImage));
    },
    [dispatch, initialImage]
  );

  const handleReset = useCallback(() => {
    dispatch(clearInitialImage());
  }, [dispatch]);

  const handleUpload = useCallback(() => {
    openUploader();
  }, [openUploader]);

  return (
    <Flex
      sx={{
        flexDir: 'column',
        width: 'full',
        height: 'full',
        position: 'absolute',
        alignItems: 'center',
        justifyContent: 'center',
        p: 4,
        gap: 4,
      }}
    >
      <Flex
        sx={{
          w: 'full',
          flexWrap: 'wrap',
          justifyContent: 'center',
          alignItems: 'center',
          gap: 2,
        }}
      >
        <Text
          sx={{
            color: 'base.200',
            fontWeight: 600,
            fontSize: 'sm',
            userSelect: 'none',
          }}
        >
          Initial Image
        </Text>
        <Spacer />
        <IAIIconButton
          tooltip="Upload Initial Image"
          aria-label="Upload Initial Image"
          icon={<FaUpload />}
          onClick={handleUpload}
          {...getUploadButtonProps()}
        />
        <IAIIconButton
          tooltip="Reset Initial Image"
          aria-label="Reset Initial Image"
          icon={<FaUndo />}
          onClick={handleReset}
          isDisabled={!initialImage}
        />
      </Flex>
      <IAIDndImage
        image={image}
        onDrop={handleDrop}
        fallback={<IAIImageLoadingFallback sx={{ bg: 'none' }} />}
        isUploadDisabled={true}
        fitContainer
      />
      <input {...getUploadInputProps()} />
    </Flex>
  );
};

export default InitialImagePreview;
