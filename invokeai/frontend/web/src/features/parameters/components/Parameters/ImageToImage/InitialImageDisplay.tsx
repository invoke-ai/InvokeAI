import { Flex, Spacer, Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIIconButton from 'common/components/IAIIconButton';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { clearInitialImage } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { FaUndo, FaUpload } from 'react-icons/fa';
import InitialImage from './InitialImage';
import { PostUploadAction } from 'services/api/types';

const selector = createSelector(
  [stateSelector],
  (state) => {
    const { initialImage } = state.generation;
    return {
      isResetButtonDisabled: !initialImage,
    };
  },
  defaultSelectorOptions
);

const postUploadAction: PostUploadAction = {
  type: 'SET_INITIAL_IMAGE',
};

const InitialImageDisplay = () => {
  const { isResetButtonDisabled } = useAppSelector(selector);
  const dispatch = useAppDispatch();

  const { getUploadButtonProps, getUploadInputProps } = useImageUploadButton({
    postUploadAction,
  });

  const handleReset = useCallback(() => {
    dispatch(clearInitialImage());
  }, [dispatch]);

  return (
    <Flex
      layerStyle="first"
      sx={{
        position: 'relative',
        flexDirection: 'column',
        height: 'full',
        width: 'full',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 'base',
        p: 2,
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
            ps: 2,
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
        <IAIIconButton
          tooltip="Upload Initial Image"
          aria-label="Upload Initial Image"
          icon={<FaUpload />}
          {...getUploadButtonProps()}
        />
        <IAIIconButton
          tooltip="Reset Initial Image"
          aria-label="Reset Initial Image"
          icon={<FaUndo />}
          onClick={handleReset}
          isDisabled={isResetButtonDisabled}
        />
      </Flex>
      <InitialImage />
      <input {...getUploadInputProps()} />
    </Flex>
  );
};

export default memo(InitialImageDisplay);
