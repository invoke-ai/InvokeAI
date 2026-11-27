import { useStore } from '@nanostores/react';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery, usePruneQueueMutation } from 'services/api/endpoints/queue';
import { $isConnected } from 'services/events/stores';

export const usePruneQueue = () => {
  const { t } = useTranslation();
  const isConnected = useStore($isConnected);
  const finishedCount = useFinishedCount();
  const [_trigger, { isLoading }] = usePruneQueueMutation({
    fixedCacheKey: 'pruneQueue',
  });

  const trigger = useCallback(async () => {
    try {
      const data = await _trigger().unwrap();
      toast({
        id: 'PRUNE_SUCCEEDED',
        title: t('queue.pruneSucceeded', { item_count: data.deleted }),
        status: 'success',
      });
    } catch {
      toast({
        id: 'PRUNE_FAILED',
        title: t('queue.pruneFailed'),
        status: 'error',
      });
    }
  }, [_trigger, t]);

  return { trigger, isLoading, isDisabled: !isConnected || !finishedCount };
};

export const useFinishedCount = () => {
  const { finishedCount } = useGetQueueStatusQuery(undefined, {
    selectFromResult: ({ data }) => {
      if (!data) {
        return { finishedCount: 0 };
      }

      return {
        finishedCount: data.queue.completed + data.queue.canceled + data.queue.failed,
      };
    },
  });

  return finishedCount;
};
