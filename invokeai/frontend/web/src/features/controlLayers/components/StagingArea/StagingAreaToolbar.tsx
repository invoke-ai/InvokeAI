import { Button, ButtonGroup, IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { INTERACTION_SCOPES } from 'common/hooks/interactionScopes';
import {
  $shouldShowStagedImage,
  sessionNextStagedImageSelected,
  sessionPrevStagedImageSelected,
  sessionStagedImageDiscarded,
  sessionStagingAreaImageAccepted,
  sessionStagingAreaReset,
} from 'features/controlLayers/store/canvasV2Slice';
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

export const StagingAreaToolbar = memo(() => {
  const dispatch = useAppDispatch();
  const session = useAppSelector((s) => s.canvasV2.session);
  const shouldShowStagedImage = useStore($shouldShowStagedImage);
  const images = useMemo(() => session.stagedImages, [session]);
  const selectedImage = useMemo(() => {
    return images[session.selectedStagedImageIndex] ?? null;
  }, [images, session.selectedStagedImageIndex]);
  const isCanvasActive = useStore(INTERACTION_SCOPES.canvas.$isActive);
  // const [changeIsImageIntermediate] = useChangeImageIsIntermediateMutation();

  const { t } = useTranslation();

  const onPrev = useCallback(() => {
    dispatch(sessionPrevStagedImageSelected());
  }, [dispatch]);

  const onNext = useCallback(() => {
    dispatch(sessionNextStagedImageSelected());
  }, [dispatch]);

  const onAccept = useCallback(() => {
    if (!selectedImage) {
      return;
    }
    dispatch(sessionStagingAreaImageAccepted({ index: session.selectedStagedImageIndex }));
  }, [dispatch, selectedImage, session.selectedStagedImageIndex]);

  const onDiscardOne = useCallback(() => {
    if (!selectedImage) {
      return;
    }
    if (images.length === 1) {
      dispatch(sessionStagingAreaReset());
    } else {
      dispatch(sessionStagedImageDiscarded({ index: session.selectedStagedImageIndex }));
    }
  }, [selectedImage, images.length, dispatch, session.selectedStagedImageIndex]);

  const onDiscardAll = useCallback(() => {
    dispatch(sessionStagingAreaReset());
  }, [dispatch]);

  const onToggleShouldShowStagedImage = useCallback(() => {
    $shouldShowStagedImage.set(!shouldShowStagedImage);
  }, [shouldShowStagedImage]);

  const onSaveStagingImage = useCallback(
    () => {
      // if (!imageDTO) {
      //   return;
      // }
      // changeIsImageIntermediate({ imageDTO, is_intermediate: false });
    },
    [
      // changeIsImageIntermediate,
      // imageDTO
    ]
  );

  useHotkeys(
    ['left'],
    onPrev,
    {
      preventDefault: true,
      enabled: isCanvasActive,
    },
    [isCanvasActive]
  );

  useHotkeys(
    ['right'],
    onNext,
    {
      preventDefault: true,
      enabled: isCanvasActive,
    },
    [isCanvasActive]
  );

  useHotkeys(
    ['enter'],
    onAccept,
    {
      preventDefault: true,
      enabled: isCanvasActive,
    },
    [isCanvasActive]
  );

  const counterText = useMemo(() => {
    if (images.length > 0) {
      return `${(session.selectedStagedImageIndex ?? 0) + 1} of ${images.length}`;
    } else {
      return `0 of 0`;
    }
  }, [images.length, session.selectedStagedImageIndex]);

  return (
    <>
      <ButtonGroup borderRadius="base" shadow="dark-lg">
        <IconButton
          tooltip={`${t('unifiedCanvas.previous')} (Left)`}
          aria-label={`${t('unifiedCanvas.previous')} (Left)`}
          icon={<PiArrowLeftBold />}
          onClick={onPrev}
          colorScheme="invokeBlue"
          isDisabled={images.length <= 1 || !shouldShowStagedImage}
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
          isDisabled={images.length <= 1 || !shouldShowStagedImage}
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
