import { useStore } from '@nanostores/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { listCursorChanged, listPriorityChanged } from 'features/queue/store/queueSlice';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useClearQueueMutation, useGetQueueStatusQuery } from 'services/api/endpoints/queue';
import { $isConnected } from 'services/events/stores';

export const useClearQueue = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { data: queueStatus } = useGetQueueStatusQuery();
  const isConnected = useStore($isConnected);
  const [_trigger, { isLoading }] = useClearQueueMutation({
    fixedCacheKey: 'clearQueue',
  });

  const trigger = useCallback(async () => {
    if (!queueStatus?.queue.total) {
      return;
    }

    try {
      await _trigger().unwrap();
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
  }, [queueStatus?.queue.total, _trigger, dispatch, t]);

  return { trigger, isLoading, isDisabled: !isConnected || !queueStatus?.queue.total };
};
