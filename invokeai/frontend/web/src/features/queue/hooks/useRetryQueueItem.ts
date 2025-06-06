import { useStore } from '@nanostores/react';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useRetryItemsByIdMutation } from 'services/api/endpoints/queue';
import { $isConnected } from 'services/events/stores';

export const useRetryQueueItem = () => {
  const isConnected = useStore($isConnected);
  const [_trigger, { isLoading }] = useRetryItemsByIdMutation();
  const { t } = useTranslation();
  const trigger = useCallback(
    async (item_id: number) => {
      try {
        const result = await _trigger([item_id]).unwrap();
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
    },
    [t, _trigger]
  );

  return { trigger, isLoading, isDisabled: !isConnected };
};
