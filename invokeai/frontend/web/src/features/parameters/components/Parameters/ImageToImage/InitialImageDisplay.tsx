import { Flex, Spacer, Text } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { useRecallParameters } from 'features/parameters/hooks/useRecallParameters';
import { clearInitialImage } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaRulerVertical, FaUndo, FaUpload } from 'react-icons/fa';
import { PostUploadAction } from 'services/api/types';
import InitialImage from './InitialImage';

const selector = createMemoizedSelector([stateSelector], (state) => {
  const { initialImage } = state.generation;
  return {
    isResetButtonDisabled: !initialImage,
    initialImage,
  };
});

const postUploadAction: PostUploadAction = {
  type: 'SET_INITIAL_IMAGE',
};

const InitialImageDisplay = () => {
  const { recallWidthAndHeight } = useRecallParameters();
  const { t } = useTranslation();
  const { isResetButtonDisabled, initialImage } = useAppSelector(selector);
  const dispatch = useAppDispatch();

  const { getUploadButtonProps, getUploadInputProps } = useImageUploadButton({
    postUploadAction,
  });

  const handleReset = useCallback(() => {
    dispatch(clearInitialImage());
  }, [dispatch]);

  const handleUseSizeInitialImage = useCallback(() => {
    if (initialImage) {
      recallWidthAndHeight(initialImage.width, initialImage.height);
    }
  }, [initialImage, recallWidthAndHeight]);

  useHotkeys('shift+d', handleUseSizeInitialImage, [initialImage]);

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
          {t('metadata.initImage')}
        </Text>
        <Spacer />
        <IAIIconButton
          tooltip="Upload Initial Image"
          aria-label="Upload Initial Image"
          icon={<FaUpload />}
          {...getUploadButtonProps()}
        />
        <IAIIconButton
          tooltip={`${t('parameters.useSize')} (Shift+D)`}
          aria-label={`${t('parameters.useSize')} (Shift+D)`}
          icon={<FaRulerVertical />}
          onClick={handleUseSizeInitialImage}
          isDisabled={isResetButtonDisabled}
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
