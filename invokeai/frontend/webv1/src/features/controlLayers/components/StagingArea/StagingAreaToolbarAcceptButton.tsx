import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useIsRegionFocused } from 'common/hooks/focus';
import { useStagingAreaContext } from 'features/controlLayers/components/StagingArea/context';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCancelQueueItemsByDestination } from 'features/queue/hooks/useCancelQueueItemsByDestination';
import { memo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiCheckBold } from 'react-icons/pi';

export const StagingAreaToolbarAcceptButton = memo(() => {
  const ctx = useStagingAreaContext();
  const canvasManager = useCanvasManager();
  const shouldShowStagedImage = useStore(canvasManager.stagingArea.$shouldShowStagedImage);
  const isCanvasFocused = useIsRegionFocused('canvas');
  const cancelQueueItemsByDestination = useCancelQueueItemsByDestination();
  const acceptSelectedIsEnabled = useStore(ctx.$acceptSelectedIsEnabled);

  const { t } = useTranslation();

  useHotkeys(
    ['enter'],
    ctx.acceptSelected,
    {
      preventDefault: true,
      enabled: isCanvasFocused && shouldShowStagedImage && acceptSelectedIsEnabled,
    },
    [ctx.acceptSelected, isCanvasFocused, shouldShowStagedImage, acceptSelectedIsEnabled]
  );

  return (
    <IconButton
      tooltip={`${t('common.accept')} (Enter)`}
      aria-label={`${t('common.accept')} (Enter)`}
      icon={<PiCheckBold />}
      onClick={ctx.acceptSelected}
      colorScheme="invokeBlue"
      isDisabled={!acceptSelectedIsEnabled || !shouldShowStagedImage || cancelQueueItemsByDestination.isDisabled}
      isLoading={cancelQueueItemsByDestination.isLoading}
    />
  );
});

StagingAreaToolbarAcceptButton.displayName = 'StagingAreaToolbarAcceptButton';
