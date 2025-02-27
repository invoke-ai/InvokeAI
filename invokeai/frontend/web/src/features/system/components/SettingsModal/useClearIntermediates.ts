import { toast } from 'features/toast/toast';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useClearIntermediatesMutation, useGetIntermediatesCountQuery } from 'services/api/endpoints/images';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

type UseClearIntermediatesReturn = {
  intermediatesCount: number | undefined;
  clearIntermediates: () => void;
  isLoading: boolean;
  hasPendingItems: boolean;
  refetchIntermediatesCount: () => void;
};

export const useClearIntermediates = (shouldShowClearIntermediates: boolean): UseClearIntermediatesReturn => {
  const { t } = useTranslation();

  const { data: intermediatesCount, refetch: refetchIntermediatesCount } = useGetIntermediatesCountQuery(undefined, {
    refetchOnMountOrArgChange: true,
    skip: !shouldShowClearIntermediates,
  });

  const [_clearIntermediates, { isLoading }] = useClearIntermediatesMutation();

  const { data: queueStatus } = useGetQueueStatusQuery();
  const hasPendingItems = useMemo(
    () => Boolean(queueStatus && (queueStatus.queue.in_progress > 0 || queueStatus.queue.pending > 0)),
    [queueStatus]
  );

  const clearIntermediates = useCallback(() => {
    if (hasPendingItems) {
      return;
    }

    _clearIntermediates()
      .unwrap()
      .then((clearedCount) => {
        // TODO(psyche): Do we need to reset things w/ canvas v2?
        // dispatch(controlAdaptersReset());
        // dispatch(resetCanvas());
        toast({
          id: 'INTERMEDIATES_CLEARED',
          title: t('settings.intermediatesCleared', { count: clearedCount }),
          status: 'info',
        });
      })
      .catch(() => {
        toast({
          id: 'INTERMEDIATES_CLEAR_FAILED',
          title: t('settings.intermediatesClearedFailed'),
          status: 'error',
        });
      });
  }, [t, _clearIntermediates, hasPendingItems]);

  return { intermediatesCount, clearIntermediates, isLoading, hasPendingItems, refetchIntermediatesCount };
};
