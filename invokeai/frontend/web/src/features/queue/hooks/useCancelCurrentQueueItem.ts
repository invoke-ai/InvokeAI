import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { isNil } from 'lodash-es';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useCancelQueueItemMutation, useGetQueueStatusQuery } from 'services/api/endpoints/queue';

export const useCancelCurrentQueueItem = () => {
  const isConnected = useAppSelector((s) => s.system.isConnected);
  const { data: queueStatus } = useGetQueueStatusQuery();
  const [trigger, { isLoading }] = useCancelQueueItemMutation();
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const currentQueueItemId = useMemo(() => queueStatus?.queue.item_id, [queueStatus?.queue.item_id]);
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

  const isDisabled = useMemo(() => !isConnected || isNil(currentQueueItemId), [isConnected, currentQueueItemId]);

  return {
    cancelQueueItem,
    isLoading,
    currentQueueItemId,
    isDisabled,
  };
};
