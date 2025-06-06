import { useStore } from '@nanostores/react';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useCancelByBatchIdsMutation } from 'services/api/endpoints/queue';
import { $isConnected } from 'services/events/stores';

export const useCancelBatch = () => {
  const isConnected = useStore($isConnected);
  const [_trigger, { isLoading }] = useCancelByBatchIdsMutation({
    fixedCacheKey: 'cancelByBatchIds',
  });
  const { t } = useTranslation();
  const trigger = useCallback(
    async (batch_id: string) => {
      try {
        await _trigger({ batch_ids: [batch_id] }).unwrap();
        toast({
          id: 'CANCEL_BATCH_SUCCEEDED',
          title: t('queue.cancelBatchSucceeded'),
          status: 'success',
        });
      } catch {
        toast({
          id: 'CANCEL_BATCH_FAILED',
          title: t('queue.cancelBatchFailed'),
          status: 'error',
        });
      }
    },
    [t, _trigger]
  );

  return { trigger, isLoading, isDisabled: !isConnected };
};
