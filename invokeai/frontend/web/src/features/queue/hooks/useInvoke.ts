import { useStore } from '@nanostores/react';
import { enqueueRequestedCanvas } from 'app/store/middleware/listenerMiddleware/listeners/enqueueRequestedLinear';
import { enqueueRequestedWorkflows } from 'app/store/middleware/listenerMiddleware/listeners/enqueueRequestedNodes';
import { enqueueRequestedUpscaling } from 'app/store/middleware/listenerMiddleware/listeners/enqueueRequestedUpscale';
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

  const enqueue = useCallback(
    (prepend: boolean, isApiValidationRun: boolean) => {
      if (!isReady) {
        return;
      }

      if (tabName === 'workflows') {
        dispatch(enqueueRequestedWorkflows({ prepend, isApiValidationRun }));
        return;
      }

      if (tabName === 'upscaling') {
        dispatch(enqueueRequestedUpscaling({ prepend }));
        return;
      }

      if (tabName === 'canvas') {
        dispatch(enqueueRequestedCanvas({ prepend }));
        return;
      }

      // Else we are not on a generation tab and should not queue
    },
    [dispatch, isReady, tabName]
  );

  const enqueueBack = useCallback(() => {
    enqueue(false, false);
  }, [enqueue]);

  const enqueueFront = useCallback(() => {
    enqueue(true, false);
  }, [enqueue]);

  return { enqueueBack, enqueueFront, isLoading, isDisabled: !isReady, enqueue };
};
