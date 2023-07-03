import { Flex, Spacer, Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { clearInitialImage } from 'features/parameters/store/generationSlice';
import { useCallback, useMemo } from 'react';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { skipToken } from '@reduxjs/toolkit/dist/query';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaLayerGroup, FaUndo, FaUpload } from 'react-icons/fa';
import useImageUploader from 'common/hooks/useImageUploader';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import IAIButton from 'common/components/IAIButton';
import { stateSelector } from 'app/store/store';
import {
  asInitialImageToggled,
  batchReset,
} from 'features/batch/store/batchSlice';
import BatchImageContainer from 'features/batch/components/BatchImageContainer';
import { PostUploadAction } from 'services/api/thunks/image';
import InitialImage from './InitialImage';

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

const InitialImageDisplay = () => {
  const { initialImage, useBatchAsInitialImage, isResetButtonDisabled } =
    useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { openUploader } = useImageUploader();

  const {
    currentData: imageDTO,
    isLoading,
    isError,
    isSuccess,
  } = useGetImageDTOQuery(initialImage?.imageName ?? skipToken);

  const postUploadAction = useMemo<PostUploadAction>(
    () =>
      useBatchAsInitialImage
        ? { type: 'ADD_TO_BATCH' }
        : { type: 'SET_INITIAL_IMAGE' },
    [useBatchAsInitialImage]
  );

  const { getUploadButtonProps, getUploadInputProps } = useImageUploadButton({
    postUploadAction,
  });

  const handleReset = useCallback(() => {
    if (useBatchAsInitialImage) {
      dispatch(batchReset());
    } else {
      dispatch(clearInitialImage());
    }
  }, [dispatch, useBatchAsInitialImage]);

  const handleUpload = useCallback(() => {
    openUploader();
  }, [openUploader]);

  const handleClickUseBatch = useCallback(() => {
    dispatch(asInitialImageToggled());
  }, [dispatch]);

  return (
    <Flex
      layerStyle={'first'}
      sx={{
        position: 'relative',
        flexDirection: 'column',
        height: 'full',
        width: 'full',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 'base',
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
            fontWeight: 600,
            userSelect: 'none',
            color: 'base.700',
            _dark: {
              color: 'base.200',
            },
          }}
        >
          Initial Image
        </Text>
        <Spacer />
        {/* <IAIButton
          tooltip={useBatchAsInitialImage ? 'Disable Batch' : 'Enable Batch'}
          aria-label={useBatchAsInitialImage ? 'Disable Batch' : 'Enable Batch'}
          leftIcon={<FaLayerGroup />}
          isChecked={useBatchAsInitialImage}
          onClick={handleClickUseBatch}
        >
          {useBatchAsInitialImage ? 'Batch' : 'Single'}
        </IAIButton> */}
        <IAIIconButton
          tooltip={
            useBatchAsInitialImage ? 'Upload to Batch' : 'Upload Initial Image'
          }
          aria-label={
            useBatchAsInitialImage ? 'Upload to Batch' : 'Upload Initial Image'
          }
          icon={<FaUpload />}
          onClick={handleUpload}
          {...getUploadButtonProps()}
        />
        <IAIIconButton
          tooltip={
            useBatchAsInitialImage ? 'Reset Batch' : 'Reset Initial Image'
          }
          aria-label={
            useBatchAsInitialImage ? 'Reset Batch' : 'Reset Initial Image'
          }
          icon={<FaUndo />}
          onClick={handleReset}
          isDisabled={isResetButtonDisabled}
        />
      </Flex>
      <InitialImage />
      {/* {useBatchAsInitialImage ? <BatchImageContainer /> : <InitialImage />} */}
      <input {...getUploadInputProps()} />
    </Flex>
  );
};

export default InitialImageDisplay;
