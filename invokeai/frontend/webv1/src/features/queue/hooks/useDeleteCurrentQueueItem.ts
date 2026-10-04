import { useCurrentQueueItemId } from 'features/queue/hooks/useCurrentQueueItemId';
import { useDeleteQueueItem } from 'features/queue/hooks/useDeleteQueueItem';
import { useCallback } from 'react';

export const useDeleteCurrentQueueItem = () => {
  const currentQueueItemId = useCurrentQueueItemId();
  const deleteQueueItem = useDeleteQueueItem();
  const trigger = useCallback(() => {
    if (currentQueueItemId === null) {
      return;
    }
    deleteQueueItem.trigger(currentQueueItemId);
  }, [currentQueueItemId, deleteQueueItem]);

  return {
    trigger,
    isLoading: deleteQueueItem.isLoading,
    isDisabled: deleteQueueItem.isDisabled || currentQueueItemId === null,
  };
};
