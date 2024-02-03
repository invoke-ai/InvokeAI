import { useAppDispatch } from 'app/store/storeHooks';
import { resetCanvas } from 'features/canvas/store/canvasSlice';
import { controlAdaptersReset } from 'features/controlAdapters/store/controlAdaptersSlice';
import { addToast } from 'features/system/store/systemSlice';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useClearIntermediatesMutation, useGetIntermediatesCountQuery } from 'services/api/endpoints/images';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

export type UseClearIntermediatesReturn = {
  intermediatesCount: number | undefined;
  clearIntermediates: () => void;
  isLoading: boolean;
  hasPendingItems: boolean;
  refetchIntermediatesCount: () => void;
};

export const useClearIntermediates = (shouldShowClearIntermediates: boolean): UseClearIntermediatesReturn => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

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
        dispatch(controlAdaptersReset());
        dispatch(resetCanvas());
        dispatch(
          addToast({
            title: t('settings.intermediatesCleared', { count: clearedCount }),
            status: 'info',
          })
        );
      })
      .catch(() => {
        dispatch(
          addToast({
            title: t('settings.intermediatesClearedFailed'),
            status: 'error',
          })
        );
      });
  }, [t, _clearIntermediates, dispatch, hasPendingItems]);

  return { intermediatesCount, clearIntermediates, isLoading, hasPendingItems, refetchIntermediatesCount };
};
