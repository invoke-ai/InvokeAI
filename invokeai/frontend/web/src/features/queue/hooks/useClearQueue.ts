import { useStore } from '@nanostores/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { listCursorChanged, listPriorityChanged } from 'features/queue/store/queueSlice';
import { toast } from 'features/toast/toast';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useClearQueueMutation, useGetQueueStatusQuery } from 'services/api/endpoints/queue';
import { $isConnected } from 'services/events/stores';

export const useClearQueue = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { data: queueStatus } = useGetQueueStatusQuery();
  const isConnected = useStore($isConnected);
  const [trigger, { isLoading }] = useClearQueueMutation({
    fixedCacheKey: 'clearQueue',
  });

  const clearQueue = useCallback(async () => {
    if (!queueStatus?.queue.total) {
      return;
    }

    try {
      await trigger().unwrap();
      toast({
        id: 'QUEUE_CLEAR_SUCCEEDED',
        title: t('queue.clearSucceeded'),
        status: 'success',
      });
      dispatch(listCursorChanged(undefined));
      dispatch(listPriorityChanged(undefined));
    } catch {
      toast({
        id: 'QUEUE_CLEAR_FAILED',
        title: t('queue.clearFailed'),
        status: 'error',
      });
    }
  }, [queueStatus?.queue.total, trigger, dispatch, t]);

  const isDisabled = useMemo(() => !isConnected || !queueStatus?.queue.total, [isConnected, queueStatus?.queue.total]);

  return {
    clearQueue,
    isLoading,
    queueStatus,
    isDisabled,
  };
};
