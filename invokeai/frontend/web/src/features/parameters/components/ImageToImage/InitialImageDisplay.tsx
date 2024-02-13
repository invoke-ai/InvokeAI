import { Flex, IconButton, Spacer, Text } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { useRecallParameters } from 'features/parameters/hooks/useRecallParameters';
import { clearInitialImage, selectGenerationSlice } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiRulerBold, PiUploadSimpleBold } from 'react-icons/pi';
import type { PostUploadAction } from 'services/api/types';

import InitialImage from './InitialImage';

const selectInitialImage = createMemoizedSelector(selectGenerationSlice, (generation) => generation.initialImage);

const postUploadAction: PostUploadAction = {
  type: 'SET_INITIAL_IMAGE',
};

const InitialImageDisplay = () => {
  const { recallWidthAndHeight } = useRecallParameters();
  const { t } = useTranslation();
  const initialImage = useAppSelector(selectInitialImage);
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
      <Flex w="full" flexWrap="wrap" justifyContent="center" alignItems="center" gap={2}>
        <Text ps={2} fontWeight="semibold" userSelect="none" color="base.200">
          {t('metadata.initImage')}
        </Text>
        <Spacer />
        <IconButton
          tooltip={t('toast.uploadInitialImage')}
          aria-label={t('toast.uploadInitialImage')}
          icon={<PiUploadSimpleBold />}
          {...getUploadButtonProps()}
        />
        <IconButton
          tooltip={`${t('parameters.useSize')} (Shift+D)`}
          aria-label={`${t('parameters.useSize')} (Shift+D)`}
          icon={<PiRulerBold />}
          onClick={handleUseSizeInitialImage}
          isDisabled={!initialImage}
        />
        <IconButton
          tooltip={t('toast.resetInitialImage')}
          aria-label={t('toast.resetInitialImage')}
          icon={<PiArrowCounterClockwiseBold />}
          onClick={handleReset}
          isDisabled={!initialImage}
        />
      </Flex>
      <InitialImage />
      <input {...getUploadInputProps()} />
    </Flex>
  );
};

export default memo(InitialImageDisplay);
