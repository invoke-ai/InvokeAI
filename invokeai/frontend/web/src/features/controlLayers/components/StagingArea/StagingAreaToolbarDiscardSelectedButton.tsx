import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useStagingAreaContext } from 'features/controlLayers/components/SimpleSession/context2';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCancelQueueItem } from 'features/queue/hooks/useCancelQueueItem';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

export const StagingAreaToolbarDiscardSelectedButton = memo(() => {
  const canvasManager = useCanvasManager();
  const shouldShowStagedImage = useStore(canvasManager.stagingArea.$shouldShowStagedImage);

  const ctx = useStagingAreaContext();
  const cancelQueueItem = useCancelQueueItem();
  const discardSelectedIsEnabled = useStore(ctx.$discardSelectedIsEnabled);

  const { t } = useTranslation();

  return (
    <IconButton
      tooltip={t('controlLayers.stagingArea.discard')}
      aria-label={t('controlLayers.stagingArea.discard')}
      icon={<PiXBold />}
      onClick={ctx.discardSelected}
      colorScheme="invokeBlue"
      isDisabled={!discardSelectedIsEnabled || cancelQueueItem.isDisabled || !shouldShowStagedImage}
      isLoading={cancelQueueItem.isLoading}
    />
  );
});

StagingAreaToolbarDiscardSelectedButton.displayName = 'StagingAreaToolbarDiscardSelectedButton';
