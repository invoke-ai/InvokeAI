import { useAppSelector } from 'app/store/storeHooks';
import { toast } from 'features/toast/toast';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery, usePauseProcessorMutation } from 'services/api/endpoints/queue';

export const usePauseProcessor = () => {
  const { t } = useTranslation();
  const isConnected = useAppSelector((s) => s.system.isConnected);
  const { data: queueStatus } = useGetQueueStatusQuery();
  const [trigger, { isLoading }] = usePauseProcessorMutation({
    fixedCacheKey: 'pauseProcessor',
  });

  const isStarted = useMemo(() => Boolean(queueStatus?.processor.is_started), [queueStatus?.processor.is_started]);

  const pauseProcessor = useCallback(async () => {
    if (!isStarted) {
      return;
    }
    try {
      await trigger().unwrap();
      toast({
        id: 'PAUSE_SUCCEEDED',
        title: t('queue.pauseSucceeded'),
        status: 'success',
      });
    } catch {
      toast({
        id: 'PAUSE_FAILED',
        title: t('queue.pauseFailed'),
        status: 'error',
      });
    }
  }, [isStarted, trigger, t]);

  const isDisabled = useMemo(() => !isConnected || !isStarted, [isConnected, isStarted]);

  return { pauseProcessor, isLoading, isStarted, isDisabled };
};
