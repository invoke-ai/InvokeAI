import { useStore } from '@nanostores/react';
import { toast } from 'features/toast/toast';
import { isNil } from 'lodash-es';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useCancelQueueItemMutation, useGetQueueStatusQuery } from 'services/api/endpoints/queue';
import { $isConnected } from 'services/events/stores';

export const useCancelCurrentQueueItem = () => {
  const isConnected = useStore($isConnected);
  const { data: queueStatus } = useGetQueueStatusQuery();
  const [trigger, { isLoading }] = useCancelQueueItemMutation();
  const { t } = useTranslation();
  const currentQueueItemId = useMemo(() => queueStatus?.queue.item_id, [queueStatus?.queue.item_id]);
  const cancelQueueItem = useCallback(async () => {
    if (!currentQueueItemId) {
      return;
    }
    try {
      await trigger(currentQueueItemId).unwrap();
      toast({
        id: 'QUEUE_CANCEL_SUCCEEDED',
        title: t('queue.cancelSucceeded'),
        status: 'success',
      });
    } catch {
      toast({
        id: 'QUEUE_CANCEL_FAILED',
        title: t('queue.cancelFailed'),
        status: 'error',
      });
    }
  }, [currentQueueItemId, t, trigger]);

  const isDisabled = useMemo(() => !isConnected || isNil(currentQueueItemId), [isConnected, currentQueueItemId]);

  return {
    cancelQueueItem,
    isLoading,
    currentQueueItemId,
    isDisabled,
  };
};
