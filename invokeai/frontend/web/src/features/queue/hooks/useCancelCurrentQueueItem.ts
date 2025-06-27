import { useCancelQueueItem } from 'features/queue/hooks/useCancelQueueItem';
import { useCurrentQueueItemId } from 'features/queue/hooks/useCurrentQueueItemId';
import { useCallback } from 'react';

export const useCancelCurrentQueueItem = () => {
  const currentQueueItemId = useCurrentQueueItemId();
  const cancelQueueItem = useCancelQueueItem();
  const trigger = useCallback(() => {
    if (currentQueueItemId === null) {
      return;
    }
    cancelQueueItem.trigger(currentQueueItemId);
  }, [currentQueueItemId, cancelQueueItem]);

  return {
    trigger,
    isLoading: cancelQueueItem.isLoading,
    isDisabled: cancelQueueItem.isDisabled || currentQueueItemId === null,
  };
};
