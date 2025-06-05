import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useIsRegionFocused } from 'common/hooks/focus';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectImageCount } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiArrowLeftBold } from 'react-icons/pi';

export const StagingAreaToolbarPrevButton = memo(() => {
  const ctx = useCanvasSessionContext();
  const itemCount = useStore(ctx.$itemCount);
  const dispatch = useAppDispatch();
  const canvasManager = useCanvasManager();
  const imageCount = useAppSelector(selectImageCount);
  const shouldShowStagedImage = useStore(canvasManager.stagingArea.$shouldShowStagedImage);
  const isCanvasFocused = useIsRegionFocused('canvas');

  const { t } = useTranslation();

  const selectPrev = useCallback(() => {
    ctx.selectPrev();
  }, [ctx]);

  useHotkeys(
    ['left'],
    ctx.selectPrev,
    {
      preventDefault: true,
      enabled: isCanvasFocused && shouldShowStagedImage && itemCount > 1,
    },
    [isCanvasFocused, shouldShowStagedImage, itemCount, ctx.selectPrev]
  );

  return (
    <IconButton
      tooltip={`${t('controlLayers.stagingArea.previous')} (Left)`}
      aria-label={`${t('controlLayers.stagingArea.previous')} (Left)`}
      icon={<PiArrowLeftBold />}
      onClick={selectPrev}
      colorScheme="invokeBlue"
      isDisabled={itemCount <= 1 || !shouldShowStagedImage}
    />
  );
});

StagingAreaToolbarPrevButton.displayName = 'StagingAreaToolbarPrevButton';
