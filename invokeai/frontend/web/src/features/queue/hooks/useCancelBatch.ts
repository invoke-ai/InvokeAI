import { useStore } from '@nanostores/react';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useCancelByBatchIdsMutation, useGetBatchStatusQuery } from 'services/api/endpoints/queue';
import { $isConnected } from 'services/events/stores';

export const useCancelBatch = (batch_id: string) => {
  const isConnected = useStore($isConnected);
  const { isCanceled } = useGetBatchStatusQuery(
    { batch_id },
    {
      selectFromResult: ({ data }) => {
        if (!data) {
          return { isCanceled: true };
        }

        return {
          isCanceled: data?.in_progress === 0 && data?.pending === 0,
        };
      },
    }
  );
  const [trigger, { isLoading }] = useCancelByBatchIdsMutation({
    fixedCacheKey: 'cancelByBatchIds',
  });
  const { t } = useTranslation();
  const cancelBatch = useCallback(async () => {
    if (isCanceled) {
      return;
    }
    try {
      await trigger({ batch_ids: [batch_id] }).unwrap();
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
  }, [batch_id, isCanceled, t, trigger]);

  return { cancelBatch, isLoading, isCanceled, isDisabled: !isConnected };
};
