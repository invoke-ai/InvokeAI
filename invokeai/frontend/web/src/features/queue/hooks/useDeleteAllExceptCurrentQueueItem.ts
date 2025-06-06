import { useStore } from '@nanostores/react';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useDeleteAllExceptCurrentMutation, useGetQueueStatusQuery } from 'services/api/endpoints/queue';
import { $isConnected } from 'services/events/stores';

export const useDeleteAllExceptCurrentQueueItem = () => {
  const { t } = useTranslation();
  const { data: queueStatus } = useGetQueueStatusQuery();
  const isConnected = useStore($isConnected);
  const [_trigger, { isLoading }] = useDeleteAllExceptCurrentMutation({
    fixedCacheKey: 'deleteAllExceptCurrent',
  });

  const trigger = useCallback(async () => {
    if (!queueStatus?.queue.pending) {
      return;
    }

    try {
      await _trigger().unwrap();
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
  }, [queueStatus?.queue.pending, _trigger, t]);

  return { trigger, isLoading, isDisabled: !isConnected || !queueStatus?.queue.pending } as const;
};
