import { Button, ButtonGroup, Flex, IconButton } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { stagingAreaImageSaved } from 'features/canvas/store/actions';
import {
  commitStagingAreaImage,
  discardStagedImage,
  discardStagedImages,
  nextStagingAreaImage,
  prevStagingAreaImage,
  selectCanvasSlice,
  setShouldShowStagingImage,
  setShouldShowStagingOutline,
} from 'features/canvas/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import {
  PiArrowLeftBold,
  PiArrowRightBold,
  PiCheckBold,
  PiEyeBold,
  PiEyeSlashBold,
  PiFloppyDiskBold,
  PiTrashSimpleBold,
  PiXBold,
} from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

const selector = createMemoizedSelector(selectCanvasSlice, (canvas) => {
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
    currentStagingAreaImage: images.length > 0 ? images[selectedImageIndex] : undefined,
    shouldShowStagingImage,
    shouldShowStagingOutline,
  };
});

const ClearStagingIntermediatesIconButton = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleDiscardStagingArea = useCallback(() => {
    dispatch(discardStagedImages());
  }, [dispatch]);

  const handleDiscardStagingImage = useCallback(() => {
    dispatch(discardStagedImage());
  }, [dispatch]);

  return (
    <>
      <IconButton
        tooltip={`${t('unifiedCanvas.discardCurrent')}`}
        aria-label={t('unifiedCanvas.discardCurrent')}
        icon={<PiXBold />}
        onClick={handleDiscardStagingImage}
        colorScheme="invokeBlue"
        fontSize={16}
      />
      <IconButton
        tooltip={`${t('unifiedCanvas.discardAll')} (Esc)`}
        aria-label={t('unifiedCanvas.discardAll')}
        icon={<PiTrashSimpleBold />}
        onClick={handleDiscardStagingArea}
        colorScheme="error"
        fontSize={16}
      />
    </>
  );
};

const IAICanvasStagingAreaToolbar = () => {
  const dispatch = useAppDispatch();
  const { currentStagingAreaImage, shouldShowStagingImage, currentIndex, total } = useAppSelector(selector);

  const { t } = useTranslation();

  const handleMouseOver = useCallback(() => {
    dispatch(setShouldShowStagingOutline(true));
  }, [dispatch]);

  const handleMouseOut = useCallback(() => {
    dispatch(setShouldShowStagingOutline(false));
  }, [dispatch]);

  const handlePrevImage = useCallback(() => dispatch(prevStagingAreaImage()), [dispatch]);

  const handleNextImage = useCallback(() => dispatch(nextStagingAreaImage()), [dispatch]);

  const handleAccept = useCallback(() => dispatch(commitStagingAreaImage()), [dispatch]);

  useHotkeys(['left'], handlePrevImage, {
    enabled: () => true,
    preventDefault: true,
  });

  useHotkeys(['right'], handleNextImage, {
    enabled: () => true,
    preventDefault: true,
  });

  useHotkeys(['enter'], handleAccept, {
    enabled: () => true,
    preventDefault: true,
  });

  useHotkeys(
    ['esc'],
    () => {
      handleDiscardStagingArea();
    },
    {
      preventDefault: true,
    }
  );

  const { data: imageDTO } = useGetImageDTOQuery(currentStagingAreaImage?.imageName ?? skipToken);

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

  useHotkeys(
    ['shift+s'],
    () => {
      shouldShowStagingImage && handleSaveToGallery();
    },
    {
      preventDefault: true,
    },
    [shouldShowStagingImage, handleSaveToGallery]
  );

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
      <ButtonGroup borderRadius="base" shadow="dark-lg">
        <IconButton
          tooltip={`${t('unifiedCanvas.previous')} (Left)`}
          aria-label={`${t('unifiedCanvas.previous')} (Left)`}
          icon={<PiArrowLeftBold />}
          onClick={handlePrevImage}
          colorScheme="invokeBlue"
          isDisabled={!shouldShowStagingImage}
        />
        <Button
          colorScheme="base"
          pointerEvents="none"
          isDisabled={!shouldShowStagingImage}
          minW={20}
        >{`${currentIndex + 1}/${total}`}</Button>
        <IconButton
          tooltip={`${t('unifiedCanvas.next')} (Right)`}
          aria-label={`${t('unifiedCanvas.next')} (Right)`}
          icon={<PiArrowRightBold />}
          onClick={handleNextImage}
          colorScheme="invokeBlue"
          isDisabled={!shouldShowStagingImage}
        />
      </ButtonGroup>
      <ButtonGroup borderRadius="base" shadow="dark-lg">
        <IconButton
          tooltip={`${t('unifiedCanvas.accept')} (Enter)`}
          aria-label={`${t('unifiedCanvas.accept')} (Enter)`}
          icon={<PiCheckBold />}
          onClick={handleAccept}
          colorScheme="invokeBlue"
        />
        <IconButton
          tooltip={shouldShowStagingImage ? t('unifiedCanvas.showResultsOn') : t('unifiedCanvas.showResultsOff')}
          aria-label={shouldShowStagingImage ? t('unifiedCanvas.showResultsOn') : t('unifiedCanvas.showResultsOff')}
          data-alert={!shouldShowStagingImage}
          icon={shouldShowStagingImage ? <PiEyeBold /> : <PiEyeSlashBold />}
          onClick={handleToggleShouldShowStagingImage}
          colorScheme="invokeBlue"
        />
        <IconButton
          tooltip={`${t('unifiedCanvas.saveToGallery')} (Shift+S)`}
          aria-label={t('unifiedCanvas.saveToGallery')}
          isDisabled={!imageDTO || !imageDTO.is_intermediate}
          icon={<PiFloppyDiskBold />}
          onClick={handleSaveToGallery}
          colorScheme="invokeBlue"
        />
        <ClearStagingIntermediatesIconButton />
      </ButtonGroup>
    </Flex>
  );
};

export default memo(IAICanvasStagingAreaToolbar);
