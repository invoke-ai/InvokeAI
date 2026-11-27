import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useStagingAreaContext } from 'features/controlLayers/components/StagingArea/context';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCancelQueueItemsByDestination } from 'features/queue/hooks/useCancelQueueItemsByDestination';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

export const StagingAreaToolbarDiscardAllButton = memo(() => {
  const canvasManager = useCanvasManager();
  const shouldShowStagedImage = useStore(canvasManager.stagingArea.$shouldShowStagedImage);

  const ctx = useStagingAreaContext();
  const { t } = useTranslation();
  const cancelQueueItemsByDestination = useCancelQueueItemsByDestination();

  return (
    <IconButton
      tooltip={`${t('controlLayers.stagingArea.discardAll')} (Esc)`}
      aria-label={t('controlLayers.stagingArea.discardAll')}
      icon={<PiTrashSimpleBold />}
      onClick={ctx.discardAll}
      colorScheme="error"
      isDisabled={cancelQueueItemsByDestination.isDisabled || !shouldShowStagedImage}
      isLoading={cancelQueueItemsByDestination.isLoading}
    />
  );
});

StagingAreaToolbarDiscardAllButton.displayName = 'StagingAreaToolbarDiscardAllButton';
