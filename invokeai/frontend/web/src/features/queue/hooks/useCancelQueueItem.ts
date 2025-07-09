import { useStore } from '@nanostores/react';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useCancelQueueItemMutation } from 'services/api/endpoints/queue';
import { $isConnected } from 'services/events/stores';

const DEFAULTS = {
  withToast: true,
};

export const useCancelQueueItem = () => {
  const isConnected = useStore($isConnected);
  const [_trigger, { isLoading }] = useCancelQueueItemMutation();
  const { t } = useTranslation();
  const trigger = useCallback(
    async (item_id: number, options?: { withToast?: boolean }) => {
      const { withToast } = { ...DEFAULTS, ...options };
      try {
        await _trigger({ item_id }).unwrap();
        if (withToast) {
          toast({
            id: 'QUEUE_CANCEL_SUCCEEDED',
            title: t('queue.cancelSucceeded'),
            status: 'success',
          });
        }
      } catch {
        if (withToast) {
          toast({
            id: 'QUEUE_CANCEL_FAILED',
            title: t('queue.cancelFailed'),
            status: 'error',
          });
        }
      }
    },
    [t, _trigger]
  );

  return { trigger, isLoading, isDisabled: !isConnected };
};
