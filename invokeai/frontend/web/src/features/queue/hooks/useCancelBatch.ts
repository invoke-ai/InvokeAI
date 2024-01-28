import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useCancelByBatchIdsMutation, useGetBatchStatusQuery } from 'services/api/endpoints/queue';

export const useCancelBatch = (batch_id: string) => {
  const isConnected = useAppSelector((s) => s.system.isConnected);
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
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const cancelBatch = useCallback(async () => {
    if (isCanceled) {
      return;
    }
    try {
      await trigger({ batch_ids: [batch_id] }).unwrap();
      dispatch(
        addToast({
          title: t('queue.cancelBatchSucceeded'),
          status: 'success',
        })
      );
    } catch {
      dispatch(
        addToast({
          title: t('queue.cancelBatchFailed'),
          status: 'error',
        })
      );
    }
  }, [batch_id, dispatch, isCanceled, t, trigger]);

  return { cancelBatch, isLoading, isCanceled, isDisabled: !isConnected };
};
