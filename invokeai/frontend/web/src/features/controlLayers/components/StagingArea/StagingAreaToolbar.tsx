import { Button, ButtonGroup, IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { INTERACTION_SCOPES, useScopeOnMount } from 'common/hooks/interactionScopes';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import {
  selectCanvasStagingAreaSlice,
  stagingAreaImageAccepted,
  stagingAreaNextStagedImageSelected,
  stagingAreaPrevStagedImageSelected,
  stagingAreaReset,
  stagingAreaStagedImageDiscarded,
} from 'features/controlLayers/store/canvasStagingAreaSlice';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import { memo, useCallback, useMemo } from 'react';
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
import { useAddImagesToBoardMutation, useChangeImageIsIntermediateMutation } from 'services/api/endpoints/images';

const selectStagedImageIndex = createSelector(
  selectCanvasStagingAreaSlice,
  (stagingArea) => stagingArea.selectedStagedImageIndex
);

const selectSelectedImage = createSelector(
  [selectCanvasStagingAreaSlice, selectStagedImageIndex],
  (stagingArea, index) => stagingArea.stagedImages[index] ?? null
);

const selectImageCount = createSelector(selectCanvasStagingAreaSlice, (stagingArea) => stagingArea.stagedImages.length);

export const StagingAreaToolbar = memo(() => {
  const dispatch = useAppDispatch();
  const canvasManager = useCanvasManager();
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);
  const index = useAppSelector(selectStagedImageIndex);
  const selectedImage = useAppSelector(selectSelectedImage);
  const imageCount = useAppSelector(selectImageCount);
  const shouldShowStagedImage = useStore(canvasManager.stagingArea.$shouldShowStagedImage);
  const isCanvasActive = useStore(INTERACTION_SCOPES.canvas.$isActive);
  const [addImageToBoard] = useAddImagesToBoardMutation();
  const [changeIsImageIntermediate] = useChangeImageIsIntermediateMutation();
  useScopeOnMount('stagingArea');

  const { t } = useTranslation();

  const onPrev = useCallback(() => {
    dispatch(stagingAreaPrevStagedImageSelected());
  }, [dispatch]);

  const onNext = useCallback(() => {
    dispatch(stagingAreaNextStagedImageSelected());
  }, [dispatch]);

  const onAccept = useCallback(() => {
    if (!selectedImage) {
      return;
    }
    dispatch(stagingAreaImageAccepted({ index }));
  }, [dispatch, index, selectedImage]);

  const onDiscardOne = useCallback(() => {
    if (!selectedImage) {
      return;
    }
    if (imageCount === 1) {
      dispatch(stagingAreaReset());
    } else {
      dispatch(stagingAreaStagedImageDiscarded({ index }));
    }
  }, [selectedImage, imageCount, dispatch, index]);

  const onDiscardAll = useCallback(() => {
    dispatch(stagingAreaReset());
  }, [dispatch]);

  const onToggleShouldShowStagedImage = useCallback(() => {
    canvasManager.stagingArea.$shouldShowStagedImage.set(!shouldShowStagedImage);
  }, [canvasManager.stagingArea.$shouldShowStagedImage, shouldShowStagedImage]);

  const onSaveStagingImage = useCallback(async () => {
    if (!selectedImage) {
      return;
    }
    if (autoAddBoardId !== 'none') {
      await addImageToBoard({ imageDTOs: [selectedImage.imageDTO], board_id: autoAddBoardId }).unwrap();
      // The changeIsImageIntermediate request will use the board_id on this specific imageDTO object, so we need to
      // update it before making the request - else the optimistic board updates will get out of whack.
      changeIsImageIntermediate({
        imageDTO: { ...selectedImage.imageDTO, board_id: autoAddBoardId },
        is_intermediate: false,
      });
    } else {
      changeIsImageIntermediate({
        imageDTO: selectedImage.imageDTO,
        is_intermediate: false,
      });
    }
  }, [addImageToBoard, autoAddBoardId, changeIsImageIntermediate, selectedImage]);

  useHotkeys(
    ['left'],
    onPrev,
    {
      preventDefault: true,
      enabled: isCanvasActive && shouldShowStagedImage && imageCount > 1,
    },
    [isCanvasActive, shouldShowStagedImage, imageCount]
  );

  useHotkeys(
    ['right'],
    onNext,
    {
      preventDefault: true,
      enabled: isCanvasActive && shouldShowStagedImage && imageCount > 1,
    },
    [isCanvasActive, shouldShowStagedImage, imageCount]
  );

  useHotkeys(
    ['enter'],
    onAccept,
    {
      preventDefault: true,
      enabled: isCanvasActive && shouldShowStagedImage && imageCount > 1,
    },
    [isCanvasActive, shouldShowStagedImage, imageCount]
  );

  const counterText = useMemo(() => {
    if (imageCount > 0) {
      return `${(index ?? 0) + 1} of ${imageCount}`;
    } else {
      return `0 of 0`;
    }
  }, [imageCount, index]);

  return (
    <>
      <ButtonGroup borderRadius="base" shadow="dark-lg">
        <IconButton
          tooltip={`${t('unifiedCanvas.previous')} (Left)`}
          aria-label={`${t('unifiedCanvas.previous')} (Left)`}
          icon={<PiArrowLeftBold />}
          onClick={onPrev}
          colorScheme="invokeBlue"
          isDisabled={imageCount <= 1 || !shouldShowStagedImage}
        />
        <Button colorScheme="base" pointerEvents="none" minW={28}>
          {counterText}
        </Button>
        <IconButton
          tooltip={`${t('unifiedCanvas.next')} (Right)`}
          aria-label={`${t('unifiedCanvas.next')} (Right)`}
          icon={<PiArrowRightBold />}
          onClick={onNext}
          colorScheme="invokeBlue"
          isDisabled={imageCount <= 1 || !shouldShowStagedImage}
        />
      </ButtonGroup>
      <ButtonGroup borderRadius="base" shadow="dark-lg">
        <IconButton
          tooltip={`${t('unifiedCanvas.accept')} (Enter)`}
          aria-label={`${t('unifiedCanvas.accept')} (Enter)`}
          icon={<PiCheckBold />}
          onClick={onAccept}
          colorScheme="invokeBlue"
          isDisabled={!selectedImage}
        />
        <IconButton
          tooltip={shouldShowStagedImage ? t('unifiedCanvas.showResultsOn') : t('unifiedCanvas.showResultsOff')}
          aria-label={shouldShowStagedImage ? t('unifiedCanvas.showResultsOn') : t('unifiedCanvas.showResultsOff')}
          data-alert={!shouldShowStagedImage}
          icon={shouldShowStagedImage ? <PiEyeBold /> : <PiEyeSlashBold />}
          onClick={onToggleShouldShowStagedImage}
          colorScheme="invokeBlue"
        />
        <IconButton
          tooltip={`${t('unifiedCanvas.saveToGallery')} (Shift+S)`}
          aria-label={t('unifiedCanvas.saveToGallery')}
          icon={<PiFloppyDiskBold />}
          onClick={onSaveStagingImage}
          colorScheme="invokeBlue"
          isDisabled={!selectedImage || !selectedImage.imageDTO.is_intermediate}
        />
        <IconButton
          tooltip={`${t('unifiedCanvas.discardCurrent')}`}
          aria-label={t('unifiedCanvas.discardCurrent')}
          icon={<PiXBold />}
          onClick={onDiscardOne}
          colorScheme="invokeBlue"
          fontSize={16}
          isDisabled={!selectedImage}
        />
        <IconButton
          tooltip={`${t('unifiedCanvas.discardAll')} (Esc)`}
          aria-label={t('unifiedCanvas.discardAll')}
          icon={<PiTrashSimpleBold />}
          onClick={onDiscardAll}
          colorScheme="error"
          fontSize={16}
        />
      </ButtonGroup>
    </>
  );
});

StagingAreaToolbar.displayName = 'StagingAreaToolbar';
