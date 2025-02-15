import { useStore } from '@nanostores/react';
import { enqueueRequested } from 'app/store/actions';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { $isReadyToEnqueue } from 'features/queue/store/readiness';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { useCallback } from 'react';
import { enqueueMutationFixedCacheKeyOptions, useEnqueueBatchMutation } from 'services/api/endpoints/queue';

export const useInvoke = () => {
  const dispatch = useAppDispatch();
  const tabName = useAppSelector(selectActiveTab);
  const isReady = useStore($isReadyToEnqueue);

  const [_, { isLoading }] = useEnqueueBatchMutation(enqueueMutationFixedCacheKeyOptions);
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
