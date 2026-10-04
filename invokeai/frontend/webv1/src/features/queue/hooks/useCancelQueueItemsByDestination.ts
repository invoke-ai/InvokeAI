import { useStore } from '@nanostores/react';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useCancelQueueItemsByDestinationMutation } from 'services/api/endpoints/queue';
import { $isConnected } from 'services/events/stores';

const DEFAULTS = {
  withToast: true,
};

export const useCancelQueueItemsByDestination = () => {
  const isConnected = useStore($isConnected);
  const [_trigger, { isLoading }] = useCancelQueueItemsByDestinationMutation();
  const { t } = useTranslation();
  const trigger = useCallback(
    async (destination: string, options?: { withToast?: boolean }) => {
      const { withToast } = { ...DEFAULTS, ...options };

      try {
        await _trigger({ destination }).unwrap();
        if (withToast) {
          toast({
            id: 'QUEUE_CANCEL_SUCCEEDED',
            title: t('queue.cancelSucceeded'),
            status: 'success',
          });
        }
      } catch (error) {
        if (withToast) {
          // Check if this is a 403 access denied error
          const isAccessDenied = error instanceof Object && 'status' in error && error.status === 403;
          toast({
            id: 'QUEUE_CANCEL_FAILED',
            title: isAccessDenied ? t('queue.cancelFailedAccessDenied') : t('queue.cancelFailed'),
            status: 'error',
          });
        }
      }
    },
    [t, _trigger]
  );

  return { trigger, isLoading, isDisabled: !isConnected };
};
