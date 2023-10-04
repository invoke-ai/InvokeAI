import { ButtonGroup, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import {
  commitStagingAreaImage,
  discardStagedImages,
  nextStagingAreaImage,
  prevStagingAreaImage,
  setShouldShowStagingImage,
  setShouldShowStagingOutline,
} from 'features/canvas/store/canvasSlice';

import { skipToken } from '@reduxjs/toolkit/dist/query';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIButton from 'common/components/IAIButton';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import {
  FaArrowLeft,
  FaArrowRight,
  FaCheck,
  FaEye,
  FaEyeSlash,
  FaSave,
  FaTimes,
} from 'react-icons/fa';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { stagingAreaImageSaved } from '../store/actions';

const selector = createSelector(
  [canvasSelector],
  (canvas) => {
    const {
      layerState: {
        stagingArea: { images, selectedImageIndex },
      },
      shouldShowStagingOutline,
      shouldShowStagingImage,
    } = canvas;

    return {
      currentIndex: selectedImageIndex,
      total: images.length,
      currentStagingAreaImage:
        images.length > 0 ? images[selectedImageIndex] : undefined,
      shouldShowStagingImage,
      shouldShowStagingOutline,
    };
  },
  defaultSelectorOptions
);

const IAICanvasStagingAreaToolbar = () => {
  const dispatch = useAppDispatch();
  const {
    currentStagingAreaImage,
    shouldShowStagingImage,
    currentIndex,
    total,
  } = useAppSelector(selector);

  const { t } = useTranslation();

  const handleMouseOver = useCallback(() => {
    dispatch(setShouldShowStagingOutline(true));
  }, [dispatch]);

  const handleMouseOut = useCallback(() => {
    dispatch(setShouldShowStagingOutline(false));
  }, [dispatch]);

  const handlePrevImage = useCallback(
    () => dispatch(prevStagingAreaImage()),
    [dispatch]
  );

  const handleNextImage = useCallback(
    () => dispatch(nextStagingAreaImage()),
    [dispatch]
  );

  const handleAccept = useCallback(
    () => dispatch(commitStagingAreaImage()),
    [dispatch]
  );

  useHotkeys(['left'], handlePrevImage, {
    enabled: () => true,
    preventDefault: true,
  });

  useHotkeys(['right'], handleNextImage, {
    enabled: () => true,
    preventDefault: true,
  });

  useHotkeys(['enter'], () => handleAccept, {
    enabled: () => true,
    preventDefault: true,
  });

  const { data: imageDTO } = useGetImageDTOQuery(
    currentStagingAreaImage?.imageName ?? skipToken
  );

  const handleToggleShouldShowStagingImage = useCallback(() => {
    dispatch(setShouldShowStagingImage(!shouldShowStagingImage));
  }, [dispatch, shouldShowStagingImage]);

  const handleSaveToGallery = useCallback(() => {
    if (!imageDTO) {
      return;
    }

    dispatch(
      stagingAreaImageSaved({
        imageDTO,
      })
    );
  }, [dispatch, imageDTO]);

  const handleDiscardStagingArea = useCallback(() => {
    dispatch(discardStagedImages());
  }, [dispatch]);

  if (!currentStagingAreaImage) {
    return null;
  }

  return (
    <Flex
      pos="absolute"
      bottom={4}
      gap={2}
      w="100%"
      align="center"
      justify="center"
      onMouseEnter={handleMouseOver}
      onMouseLeave={handleMouseOut}
    >
      <ButtonGroup isAttached borderRadius="base" shadow="dark-lg">
        <IAIIconButton
          tooltip={`${t('unifiedCanvas.previous')} (Left)`}
          aria-label={`${t('unifiedCanvas.previous')} (Left)`}
          icon={<FaArrowLeft />}
          onClick={handlePrevImage}
          colorScheme="accent"
          isDisabled={!shouldShowStagingImage}
        />
        <IAIButton
          colorScheme="base"
          pointerEvents="none"
          isDisabled={!shouldShowStagingImage}
          minW={20}
        >{`${currentIndex + 1}/${total}`}</IAIButton>
        <IAIIconButton
          tooltip={`${t('unifiedCanvas.next')} (Right)`}
          aria-label={`${t('unifiedCanvas.next')} (Right)`}
          icon={<FaArrowRight />}
          onClick={handleNextImage}
          colorScheme="accent"
          isDisabled={!shouldShowStagingImage}
        />
      </ButtonGroup>
      <ButtonGroup isAttached borderRadius="base" shadow="dark-lg">
        <IAIIconButton
          tooltip={`${t('unifiedCanvas.accept')} (Enter)`}
          aria-label={`${t('unifiedCanvas.accept')} (Enter)`}
          icon={<FaCheck />}
          onClick={handleAccept}
          colorScheme="accent"
        />
        <IAIIconButton
          tooltip={
            shouldShowStagingImage
              ? t('unifiedCanvas.showResultsOn')
              : t('unifiedCanvas.showResultsOff')
          }
          aria-label={
            shouldShowStagingImage
              ? t('unifiedCanvas.showResultsOn')
              : t('unifiedCanvas.showResultsOff')
          }
          data-alert={!shouldShowStagingImage}
          icon={shouldShowStagingImage ? <FaEye /> : <FaEyeSlash />}
          onClick={handleToggleShouldShowStagingImage}
          colorScheme="accent"
        />
        <IAIIconButton
          tooltip={t('unifiedCanvas.saveToGallery')}
          aria-label={t('unifiedCanvas.saveToGallery')}
          isDisabled={!imageDTO || !imageDTO.is_intermediate}
          icon={<FaSave />}
          onClick={handleSaveToGallery}
          colorScheme="accent"
        />
        <IAIIconButton
          tooltip={t('unifiedCanvas.discardAll')}
          aria-label={t('unifiedCanvas.discardAll')}
          icon={<FaTimes />}
          onClick={handleDiscardStagingArea}
          colorScheme="error"
          fontSize={20}
        />
      </ButtonGroup>
    </Flex>
  );
};

export default memo(IAICanvasStagingAreaToolbar);
