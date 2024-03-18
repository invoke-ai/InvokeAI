import { enqueueRequested } from 'app/store/actions';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useIsReadyToEnqueue } from 'common/hooks/useIsReadyToEnqueue';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useCallback, useMemo } from 'react';
import { useEnqueueBatchMutation } from 'services/api/endpoints/queue';

export const useQueueBack = () => {
  const dispatch = useAppDispatch();
  const tabName = useAppSelector(activeTabNameSelector);
  const { isReady } = useIsReadyToEnqueue();
  const [_, { isLoading }] = useEnqueueBatchMutation({
    fixedCacheKey: 'enqueueBatch',
  });
  const isDisabled = useMemo(() => !isReady, [isReady]);
  const queueBack = useCallback(() => {
    if (isDisabled) {
      return;
    }
    dispatch(enqueueRequested({ tabName, prepend: false }));
  }, [dispatch, isDisabled, tabName]);

  return { queueBack, isLoading, isDisabled };
};
