import { useStore } from '@nanostores/react';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useClearQueueMutation, useGetQueueStatusQuery } from 'services/api/endpoints/queue';
import { $isConnected } from 'services/events/stores';

export const useClearQueue = () => {
  const { t } = useTranslation();
  const { data: queueStatus } = useGetQueueStatusQuery();
  const isConnected = useStore($isConnected);
  const [_trigger, { isLoading }] = useClearQueueMutation({
    fixedCacheKey: 'clearQueue',
  });

  const trigger = useCallback(async () => {
    if (!queueStatus?.queue.total) {
      return;
    }

    try {
      await _trigger().unwrap();
      toast({
        id: 'QUEUE_CLEAR_SUCCEEDED',
        title: t('queue.clearSucceeded'),
        status: 'success',
      });
    } catch (error) {
      // Check if this is a 403 access denied error
      const isAccessDenied = error instanceof Object && 'status' in error && error.status === 403;
      toast({
        id: 'QUEUE_CLEAR_FAILED',
        title: isAccessDenied ? t('queue.clearFailedAccessDenied') : t('queue.clearFailed'),
        status: 'error',
      });
    }
  }, [queueStatus?.queue.total, _trigger, t]);

  return { trigger, isLoading, isDisabled: !isConnected || !queueStatus?.queue.total };
};
