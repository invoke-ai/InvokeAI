import { Flex, Spacer } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { InvText } from 'common/components/InvText/wrapper';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { useRecallParameters } from 'features/parameters/hooks/useRecallParameters';
import { clearInitialImage } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiRulerBold, PiUploadSimpleBold } from 'react-icons/pi'
import type { PostUploadAction } from 'services/api/types';

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
      position="relative"
      flexDirection="column"
      height="full"
      width="full"
      alignItems="center"
      justifyContent="center"
      borderRadius="base"
      p={2}
      gap={4}
    >
      <Flex
        w="full"
        flexWrap="wrap"
        justifyContent="center"
        alignItems="center"
        gap={2}
      >
        <InvText
          ps={2}
          fontWeight="semibold"
          userSelect="none"
          color="base.200"
        >
          {t('metadata.initImage')}
        </InvText>
        <Spacer />
        <InvIconButton
          tooltip="Upload Initial Image"
          aria-label="Upload Initial Image"
          icon={<PiUploadSimpleBold />}
          {...getUploadButtonProps()}
        />
        <InvIconButton
          tooltip={`${t('parameters.useSize')} (Shift+D)`}
          aria-label={`${t('parameters.useSize')} (Shift+D)`}
          icon={<PiRulerBold />}
          onClick={handleUseSizeInitialImage}
          isDisabled={isResetButtonDisabled}
        />
        <InvIconButton
          tooltip="Reset Initial Image"
          aria-label="Reset Initial Image"
          icon={<PiArrowCounterClockwiseBold />}
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
