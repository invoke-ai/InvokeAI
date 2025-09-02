import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useIsRegionFocused } from 'common/hooks/focus';
import { useStagingAreaContext } from 'features/controlLayers/components/StagingArea/context';
import { useCanvasManager } from 'features/controlLayers/hooks/useCanvasManager';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiArrowLeftBold } from 'react-icons/pi';

export const StagingAreaToolbarPrevButton = memo(() => {
  const canvasManager = useCanvasManager();
  const shouldShowStagedImage = useStore(canvasManager.stagingArea.$shouldShowStagedImage);
  const ctx = useStagingAreaContext();
  const itemCount = useStore(ctx.$itemCount);
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
