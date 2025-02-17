import { useStore } from '@nanostores/react';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useRetryItemsByIdMutation } from 'services/api/endpoints/queue';
import { $isConnected } from 'services/events/stores';

export const useRetryQueueItem = (item_id: number) => {
  const isConnected = useStore($isConnected);
  const [trigger, { isLoading }] = useRetryItemsByIdMutation();
  const { t } = useTranslation();
  const retryQueueItem = useCallback(async () => {
    try {
      const result = await trigger([item_id]).unwrap();
      if (!result.retried_item_ids.includes(item_id)) {
        throw new Error('Failed to retry item');
      }
      toast({
        id: 'QUEUE_RETRY_SUCCEEDED',
        title: t('queue.retrySucceeded'),
        status: 'success',
      });
    } catch {
      toast({
        id: 'QUEUE_RETRY_FAILED',
        title: t('queue.retryFailed'),
        status: 'error',
      });
    }
  }, [item_id, t, trigger]);

  return { retryQueueItem, isLoading, isDisabled: !isConnected };
};
