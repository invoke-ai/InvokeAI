import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useIsRegionFocused } from 'common/hooks/focus';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import {
  selectImageCount,
  stagingAreaPrevStagedImageSelected,
} from 'features/controlLayers/store/canvasStagingAreaSlice';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiArrowLeftBold } from 'react-icons/pi';

export const StagingAreaToolbarPrevButton = memo(() => {
  const dispatch = useAppDispatch();
  const canvasManager = useCanvasManager();
  const imageCount = useAppSelector(selectImageCount);
  const shouldShowStagedImage = useStore(canvasManager.stagingArea.$shouldShowStagedImage);
  const isCanvasFocused = useIsRegionFocused('canvas');

  const { t } = useTranslation();

  const selectPrev = useCallback(() => {
    dispatch(stagingAreaPrevStagedImageSelected());
  }, [dispatch]);

  useHotkeys(
    ['left'],
    selectPrev,
    {
      preventDefault: true,
      enabled: isCanvasFocused && shouldShowStagedImage && imageCount > 1,
    },
    [isCanvasFocused, shouldShowStagedImage, imageCount]
  );

  return (
    <IconButton
      tooltip={`${t('controlLayers.stagingArea.previous')} (Left)`}
      aria-label={`${t('controlLayers.stagingArea.previous')} (Left)`}
      icon={<PiArrowLeftBold />}
      onClick={selectPrev}
      colorScheme="invokeBlue"
      isDisabled={imageCount <= 1 || !shouldShowStagedImage}
    />
  );
});

StagingAreaToolbarPrevButton.displayName = 'StagingAreaToolbarPrevButton';
