import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  useClearQueueMutation,
  useGetQueueStatusQuery,
} from 'services/api/endpoints/queue';
import { listCursorChanged, listPriorityChanged } from '../store/queueSlice';

export const useClearQueue = () => {
  const { t } = useTranslation();

  const dispatch = useAppDispatch();
  const { data: queueStatus } = useGetQueueStatusQuery();

  const [trigger, { isLoading }] = useClearQueueMutation({
    fixedCacheKey: 'clearQueue',
  });

  const clearQueue = useCallback(async () => {
    if (!queueStatus?.queue.total) {
      return;
    }

    try {
      await trigger().unwrap();
      dispatch(
        addToast({
          title: t('queue.clearSucceeded'),
          status: 'success',
        })
      );
      dispatch(listCursorChanged(undefined));
      dispatch(listPriorityChanged(undefined));
    } catch {
      dispatch(
        addToast({
          title: t('queue.clearFailed'),
          status: 'error',
        })
      );
    }
  }, [queueStatus?.queue.total, trigger, dispatch, t]);

  return { clearQueue, isLoading, queueStatus };
};
