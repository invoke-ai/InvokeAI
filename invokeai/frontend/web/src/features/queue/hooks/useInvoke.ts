import { enqueueRequested } from 'app/store/actions';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useIsReadyToEnqueue } from 'common/hooks/useIsReadyToEnqueue';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { useCallback } from 'react';
import { useEnqueueBatchMutation } from 'services/api/endpoints/queue';

export const useInvoke = () => {
  const dispatch = useAppDispatch();
  const tabName = useAppSelector(selectActiveTab);
  const { isReady } = useIsReadyToEnqueue();
  const [_, { isLoading }] = useEnqueueBatchMutation({
    fixedCacheKey: 'enqueueBatch',
  });
  const queueBack = useCallback(() => {
    if (!isReady) {
      return;
    }
    dispatch(enqueueRequested({ tabName, prepend: false }));
  }, [dispatch, isReady, tabName]);
  const queueFront = useCallback(() => {
    if (!isReady) {
      return;
    }
    dispatch(enqueueRequested({ tabName, prepend: true }));
  }, [dispatch, isReady, tabName]);

  return { queueBack, queueFront, isLoading, isDisabled: !isReady };
};
