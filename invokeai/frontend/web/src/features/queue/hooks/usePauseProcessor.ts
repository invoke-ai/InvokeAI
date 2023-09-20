import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  useGetQueueStatusQuery,
  usePauseProcessorMutation,
} from 'services/api/endpoints/queue';

export const usePauseProcessor = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { data: queueStatus } = useGetQueueStatusQuery();
  const [trigger, { isLoading }] = usePauseProcessorMutation({
    fixedCacheKey: 'pauseProcessor',
  });

  const isStarted = useMemo(
    () => Boolean(queueStatus?.processor.is_started),
    [queueStatus?.processor.is_started]
  );

  const pauseProcessor = useCallback(async () => {
    if (!isStarted) {
      return;
    }
    try {
      await trigger().unwrap();
      dispatch(
        addToast({
          title: t('queue.pauseSucceeded'),
          status: 'success',
        })
      );
    } catch {
      dispatch(
        addToast({
          title: t('queue.pauseFailed'),
          status: 'error',
        })
      );
    }
  }, [isStarted, trigger, dispatch, t]);

  return { pauseProcessor, isLoading, isStarted };
};
