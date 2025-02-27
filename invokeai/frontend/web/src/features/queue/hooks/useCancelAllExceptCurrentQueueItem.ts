import { useStore } from '@nanostores/react';
import { toast } from 'features/toast/toast';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useCancelAllExceptCurrentMutation, useGetQueueStatusQuery } from 'services/api/endpoints/queue';
import { $isConnected } from 'services/events/stores';

export const useCancelAllExceptCurrentQueueItem = () => {
  const { t } = useTranslation();
  const { data: queueStatus } = useGetQueueStatusQuery();
  const isConnected = useStore($isConnected);
  const [trigger, { isLoading }] = useCancelAllExceptCurrentMutation({
    fixedCacheKey: 'cancelAllExceptCurrent',
  });

  const cancelAllExceptCurrentQueueItem = useCallback(async () => {
    if (!queueStatus?.queue.pending) {
      return;
    }

    try {
      await trigger().unwrap();
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
  }, [queueStatus?.queue.pending, trigger, t]);

  const isDisabled = useMemo(
    () => !isConnected || !queueStatus?.queue.pending,
    [isConnected, queueStatus?.queue.pending]
  );

  return {
    cancelAllExceptCurrentQueueItem,
    isLoading,
    queueStatus,
    isDisabled,
  };
};
