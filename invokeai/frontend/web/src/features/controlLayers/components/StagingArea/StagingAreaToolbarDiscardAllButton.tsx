import { IconButton } from '@invoke-ai/ui-library';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { useCancelQueueItemsByDestination } from 'features/queue/hooks/useCancelQueueItemsByDestination';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

export const StagingAreaToolbarDiscardAllButton = memo(({ isDisabled }: { isDisabled?: boolean }) => {
  const ctx = useCanvasSessionContext();
  const { t } = useTranslation();
  const cancelQueueItemsByDestination = useCancelQueueItemsByDestination();

  const discardAll = useCallback(() => {
    ctx.discardAll();
    cancelQueueItemsByDestination.trigger(ctx.session.id, { withToast: false });
  }, [cancelQueueItemsByDestination, ctx]);

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
