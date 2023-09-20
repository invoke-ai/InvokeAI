import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  useGetQueueStatusQuery,
  useResumeProcessorMutation,
} from 'services/api/endpoints/queue';

export const useResumeProcessor = () => {
  const dispatch = useAppDispatch();
  const { data: queueStatus } = useGetQueueStatusQuery();
  const { t } = useTranslation();
  const [trigger, { isLoading }] = useResumeProcessorMutation({
    fixedCacheKey: 'resumeProcessor',
  });

  const isStarted = useMemo(
    () => Boolean(queueStatus?.processor.is_started),
    [queueStatus?.processor.is_started]
  );

  const resumeProcessor = useCallback(async () => {
    if (isStarted) {
      return;
    }
    try {
      await trigger().unwrap();
      dispatch(
        addToast({
          title: t('queue.resumeSucceeded'),
          status: 'success',
        })
      );
    } catch {
      dispatch(
        addToast({
          title: t('queue.resumeFailed'),
          status: 'error',
        })
      );
    }
  }, [isStarted, trigger, dispatch, t]);

  return { resumeProcessor, isLoading, isStarted };
};
