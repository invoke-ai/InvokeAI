import { useStore } from '@nanostores/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { listCursorChanged, listPriorityChanged } from 'features/queue/store/queueSlice';
import { toast } from 'features/toast/toast';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery, usePruneQueueMutation } from 'services/api/endpoints/queue';
import { $isConnected } from 'services/events/stores';

export const usePruneQueue = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const isConnected = useStore($isConnected);
  const [trigger, { isLoading }] = usePruneQueueMutation({
    fixedCacheKey: 'pruneQueue',
  });
  const { finishedCount } = useGetQueueStatusQuery(undefined, {
    selectFromResult: ({ data }) => {
      if (!data) {
        return { finishedCount: 0 };
      }

      return {
        finishedCount: data.queue.completed + data.queue.canceled + data.queue.failed,
      };
    },
  });

  const pruneQueue = useCallback(async () => {
    if (!finishedCount) {
      return;
    }
    try {
      const data = await trigger().unwrap();
      toast({
        id: 'PRUNE_SUCCEEDED',
        title: t('queue.pruneSucceeded', { item_count: data.deleted }),
        status: 'success',
      });
      dispatch(listCursorChanged(undefined));
      dispatch(listPriorityChanged(undefined));
    } catch {
      toast({
        id: 'PRUNE_FAILED',
        title: t('queue.pruneFailed'),
        status: 'error',
      });
    }
  }, [finishedCount, trigger, dispatch, t]);

  const isDisabled = useMemo(() => !isConnected || !finishedCount, [finishedCount, isConnected]);

  return { pruneQueue, isLoading, finishedCount, isDisabled };
};
