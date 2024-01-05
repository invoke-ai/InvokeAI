import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useCancelQueueItemMutation } from 'services/api/endpoints/queue';

export const useCancelQueueItem = (item_id: number) => {
  const isConnected = useAppSelector((s) => s.system.isConnected);
  const [trigger, { isLoading }] = useCancelQueueItemMutation();
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const cancelQueueItem = useCallback(async () => {
    try {
      await trigger(item_id).unwrap();
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
  }, [dispatch, item_id, t, trigger]);

  return { cancelQueueItem, isLoading, isDisabled: !isConnected };
};
