import { useCurrentQueueItemId } from 'features/queue/hooks/useCurrentQueueItemId';
import { useCallback } from 'react';

import { useCancelQueueItem } from './useCancelQueueItem';

export const useCancelCurrentQueueItem = () => {
  const currentQueueItemId = useCurrentQueueItemId();
  const cancelQueueItem = useCancelQueueItem();
  const trigger = useCallback(
    (options?: { withToast?: boolean }) => {
      if (currentQueueItemId === null) {
        return;
      }
      cancelQueueItem.trigger(currentQueueItemId, options);
    },
    [currentQueueItemId, cancelQueueItem]
  );

  return {
    trigger,
    isLoading: cancelQueueItem.isLoading,
    isDisabled: cancelQueueItem.isDisabled || currentQueueItemId === null,
  };
};
