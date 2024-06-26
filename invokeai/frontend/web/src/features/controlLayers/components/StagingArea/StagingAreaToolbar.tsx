import { Button, ButtonGroup, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  stagingAreaImageAccepted,
  stagingAreaImageDiscarded,
  stagingAreaNextImageSelected,
  stagingAreaPreviousImageSelected,
  stagingAreaReset,
} from 'features/controlLayers/store/canvasV2Slice';
import type { CanvasV2State } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiArrowLeftBold, PiArrowRightBold, PiCheckBold, PiTrashSimpleBold, PiXBold } from 'react-icons/pi';

export const StagingAreaToolbar = memo(() => {
  const stagingArea = useAppSelector((s) => s.canvasV2.stagingArea);

  if (!stagingArea || stagingArea.images.length === 0) {
    return null;
  }

  return <StagingAreaToolbarContent stagingArea={stagingArea} />;
});

StagingAreaToolbar.displayName = 'StagingAreaToolbar';

type Props = {
  stagingArea: NonNullable<CanvasV2State['stagingArea']>;
};

export const StagingAreaToolbarContent = memo(({ stagingArea }: Props) => {
  const dispatch = useAppDispatch();
  const images = useMemo(() => stagingArea.images, [stagingArea]);
  const imageDTO = useMemo(() => {
    if (stagingArea.selectedImageIndex === null) {
      return null;
    }
    return images[stagingArea.selectedImageIndex] ?? null;
  }, [images, stagingArea.selectedImageIndex]);

  const { t } = useTranslation();

  const onPrev = useCallback(() => {
    dispatch(stagingAreaPreviousImageSelected());
  }, [dispatch]);

  const onNext = useCallback(() => {
    dispatch(stagingAreaNextImageSelected());
  }, [dispatch]);

  const onAccept = useCallback(() => {
    if (!imageDTO || !stagingArea) {
      return;
    }
    dispatch(stagingAreaImageAccepted({ imageDTO }));
  }, [dispatch, imageDTO, stagingArea]);

  const onDiscardOne = useCallback(() => {
    if (!imageDTO || !stagingArea) {
      return;
    }
    if (images.length === 1) {
      dispatch(stagingAreaReset());
    } else {
      dispatch(stagingAreaImageDiscarded({ imageDTO }));
    }
  }, [dispatch, imageDTO, images.length, stagingArea]);

  const onDiscardAll = useCallback(() => {
    if (!stagingArea) {
      return;
    }
    dispatch(stagingAreaReset());
  }, [dispatch, stagingArea]);

  useHotkeys(['left'], onPrev, {
    preventDefault: true,
  });

  useHotkeys(['right'], onNext, {
    preventDefault: true,
  });

  useHotkeys(['enter'], onAccept, {
    preventDefault: true,
  });

  return (
    <>
      <ButtonGroup borderRadius="base" shadow="dark-lg">
        <IconButton
          tooltip={`${t('unifiedCanvas.previous')} (Left)`}
          aria-label={`${t('unifiedCanvas.previous')} (Left)`}
          icon={<PiArrowLeftBold />}
          onClick={onPrev}
          colorScheme="invokeBlue"
        />
        <Button
          colorScheme="base"
          pointerEvents="none"
          minW={20}
        >{`${(stagingArea.selectedImageIndex ?? 0) + 1}/${images.length}`}</Button>
        <IconButton
          tooltip={`${t('unifiedCanvas.next')} (Right)`}
          aria-label={`${t('unifiedCanvas.next')} (Right)`}
          icon={<PiArrowRightBold />}
          onClick={onNext}
          colorScheme="invokeBlue"
        />
      </ButtonGroup>
      <ButtonGroup borderRadius="base" shadow="dark-lg">
        <IconButton
          tooltip={`${t('unifiedCanvas.accept')} (Enter)`}
          aria-label={`${t('unifiedCanvas.accept')} (Enter)`}
          icon={<PiCheckBold />}
          onClick={onAccept}
          colorScheme="invokeBlue"
        />
        {/* <IconButton
          tooltip={`${t('unifiedCanvas.saveToGallery')} (Shift+S)`}
          aria-label={t('unifiedCanvas.saveToGallery')}
          isDisabled={!imageDTO || !imageDTO.is_intermediate}
          icon={<PiFloppyDiskBold />}
          onClick={handleSaveToGallery}
          colorScheme="invokeBlue"
        /> */}
        <IconButton
          tooltip={`${t('unifiedCanvas.discardCurrent')}`}
          aria-label={t('unifiedCanvas.discardCurrent')}
          icon={<PiXBold />}
          onClick={onDiscardOne}
          colorScheme="invokeBlue"
          fontSize={16}
          isDisabled={images.length <= 1}
        />
        <IconButton
          tooltip={`${t('unifiedCanvas.discardAll')} (Esc)`}
          aria-label={t('unifiedCanvas.discardAll')}
          icon={<PiTrashSimpleBold />}
          onClick={onDiscardAll}
          colorScheme="error"
          fontSize={16}
          isDisabled={images.length === 0}
        />
      </ButtonGroup>
    </>
  );
});

StagingAreaToolbarContent.displayName = 'StagingAreaToolbarContent';
