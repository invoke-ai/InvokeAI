import { IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { selectCanvasSessionId } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { useCancelQueueItemsByDestination } from 'features/queue/hooks/useCancelQueueItemsByDestination';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

export const StagingAreaToolbarDiscardAllButton = memo(({ isDisabled }: { isDisabled?: boolean }) => {
  const ctx = useCanvasSessionContext();
  const { t } = useTranslation();
  const cancelQueueItemsByDestination = useCancelQueueItemsByDestination();
  const canvasSessionId = useAppSelector(selectCanvasSessionId);

  const discardAll = useCallback(() => {
    ctx.discardAll();
    cancelQueueItemsByDestination.trigger(canvasSessionId, { withToast: false });
  }, [cancelQueueItemsByDestination, ctx, canvasSessionId]);

  return (
    <IconButton
      tooltip={`${t('controlLayers.stagingArea.discardAll')} (Esc)`}
      aria-label={t('controlLayers.stagingArea.discardAll')}
      icon={<PiTrashSimpleBold />}
      onClick={discardAll}
      colorScheme="error"
      isDisabled={isDisabled || cancelQueueItemsByDestination.isDisabled}
      isLoading={cancelQueueItemsByDestination.isLoading}
    />
  );
});

StagingAreaToolbarDiscardAllButton.displayName = 'StagingAreaToolbarDiscardAllButton';
