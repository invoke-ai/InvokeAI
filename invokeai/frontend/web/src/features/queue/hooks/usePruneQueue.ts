import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { listCursorChanged, listPriorityChanged } from 'features/queue/store/queueSlice';
import { addToast } from 'features/system/store/systemSlice';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery, usePruneQueueMutation } from 'services/api/endpoints/queue';

export const usePruneQueue = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const isConnected = useAppSelector((s) => s.system.isConnected);
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
      dispatch(
        addToast({
          title: t('queue.pruneSucceeded', { item_count: data.deleted }),
          status: 'success',
        })
      );
      dispatch(listCursorChanged(undefined));
      dispatch(listPriorityChanged(undefined));
    } catch {
      dispatch(
        addToast({
          title: t('queue.pruneFailed'),
          status: 'error',
        })
      );
    }
  }, [finishedCount, trigger, dispatch, t]);

  const isDisabled = useMemo(() => !isConnected || !finishedCount, [finishedCount, isConnected]);

  return { pruneQueue, isLoading, finishedCount, isDisabled };
};
