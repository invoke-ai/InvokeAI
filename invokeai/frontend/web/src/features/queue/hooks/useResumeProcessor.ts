import { useStore } from '@nanostores/react';
import { toast } from 'features/toast/toast';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery, useResumeProcessorMutation } from 'services/api/endpoints/queue';
import { $isConnected } from 'services/events/stores';

export const useResumeProcessor = () => {
  const isConnected = useStore($isConnected);
  const { data: queueStatus } = useGetQueueStatusQuery();
  const { t } = useTranslation();
  const [trigger, { isLoading }] = useResumeProcessorMutation({
    fixedCacheKey: 'resumeProcessor',
  });

  const isStarted = useMemo(() => Boolean(queueStatus?.processor.is_started), [queueStatus?.processor.is_started]);

  const resumeProcessor = useCallback(async () => {
    if (isStarted) {
      return;
    }
    try {
      await trigger().unwrap();
      toast({
        id: 'PROCESSOR_RESUMED',
        title: t('queue.resumeSucceeded'),
        status: 'success',
      });
    } catch {
      toast({
        id: 'PROCESSOR_RESUME_FAILED',
        title: t('queue.resumeFailed'),
        status: 'error',
      });
    }
  }, [isStarted, trigger, t]);

  const isDisabled = useMemo(() => !isConnected || isStarted, [isConnected, isStarted]);

  return { resumeProcessor, isLoading, isStarted, isDisabled };
};
