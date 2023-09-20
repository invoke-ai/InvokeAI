import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  useCancelQueueItemMutation,
  useGetQueueStatusQuery,
} from 'services/api/endpoints/queue';

export const useCancelCurrentQueueItem = () => {
  const { data: queueStatus } = useGetQueueStatusQuery();
  const [trigger, { isLoading }] = useCancelQueueItemMutation();
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const currentQueueItemId = useMemo(
    () => queueStatus?.queue.item_id,
    [queueStatus?.queue.item_id]
  );
  const cancelQueueItem = useCallback(async () => {
    if (!currentQueueItemId) {
      return;
    }
    try {
      await trigger(currentQueueItemId).unwrap();
      dispatch(
        addToast({
          title: t('queue.cancelSucceeded'),
          status: 'success',
        })
      );
    } catch {
      dispatch(
        addToast({
          title: t('queue.cancelFailed'),
          status: 'error',
        })
      );
    }
  }, [currentQueueItemId, dispatch, t, trigger]);

  return { cancelQueueItem, isLoading, currentQueueItemId };
};
