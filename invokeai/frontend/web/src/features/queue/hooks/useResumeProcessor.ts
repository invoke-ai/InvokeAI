import { useStore } from '@nanostores/react';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery, useResumeProcessorMutation } from 'services/api/endpoints/queue';
import { $isConnected } from 'services/events/stores';

export const useResumeProcessor = () => {
  const isConnected = useStore($isConnected);
  const { data: queueStatus } = useGetQueueStatusQuery();
  const { t } = useTranslation();
  const [_trigger, { isLoading }] = useResumeProcessorMutation({
    fixedCacheKey: 'resumeProcessor',
  });

  const trigger = useCallback(async () => {
    try {
      await _trigger().unwrap();
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
  }, [_trigger, t]);

  return { trigger, isLoading, isDisabled: !isConnected || queueStatus?.processor.is_started };
};
