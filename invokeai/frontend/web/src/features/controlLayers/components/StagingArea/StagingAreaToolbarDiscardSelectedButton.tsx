import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { useCancelQueueItem } from 'features/queue/hooks/useCancelQueueItem';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

export const StagingAreaToolbarDiscardSelectedButton = memo(({ isDisabled }: { isDisabled?: boolean }) => {
  const ctx = useCanvasSessionContext();
  const cancelQueueItem = useCancelQueueItem();
  const selectedItemId = useStore(ctx.$selectedItemId);

  const { t } = useTranslation();

  const discardSelected = useCallback(async () => {
    if (selectedItemId === null) {
      return;
    }
    ctx.discard(selectedItemId);
    await cancelQueueItem.trigger(selectedItemId, { withToast: false });
  }, [selectedItemId, ctx, cancelQueueItem]);

  return (
    <IconButton
      tooltip={t('controlLayers.stagingArea.discard')}
      aria-label={t('controlLayers.stagingArea.discard')}
      icon={<PiXBold />}
      onClick={discardSelected}
      colorScheme="invokeBlue"
      isDisabled={selectedItemId === null || cancelQueueItem.isDisabled || isDisabled}
      isLoading={cancelQueueItem.isLoading}
    />
  );
});

StagingAreaToolbarDiscardSelectedButton.displayName = 'StagingAreaToolbarDiscardSelectedButton';
