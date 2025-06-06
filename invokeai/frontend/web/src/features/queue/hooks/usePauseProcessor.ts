import { useStore } from '@nanostores/react';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery, usePauseProcessorMutation } from 'services/api/endpoints/queue';
import { $isConnected } from 'services/events/stores';

export const usePauseProcessor = () => {
  const { t } = useTranslation();
  const isConnected = useStore($isConnected);
  const { data: queueStatus } = useGetQueueStatusQuery();
  const [_trigger, { isLoading }] = usePauseProcessorMutation({
    fixedCacheKey: 'pauseProcessor',
  });

  const trigger = useCallback(async () => {
    try {
      await _trigger().unwrap();
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
  }, [_trigger, t]);

  return { trigger, isLoading, isDisabled: !isConnected || !queueStatus?.processor.is_started };
};
