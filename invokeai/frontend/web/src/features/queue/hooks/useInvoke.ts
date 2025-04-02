import { useStore } from '@nanostores/react';
import { logger } from 'app/logging/logger';
import { enqueueRequestedCanvas } from 'app/store/middleware/listenerMiddleware/listeners/enqueueRequestedLinear';
import { enqueueRequestedUpscaling } from 'app/store/middleware/listenerMiddleware/listeners/enqueueRequestedUpscale';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { withResultAsync } from 'common/util/result';
import { parseify } from 'common/util/serialize';
import { useIsWorkflowEditorLocked } from 'features/nodes/hooks/useIsWorkflowEditorLocked';
import { useEnqueueWorkflows } from 'features/queue/hooks/useEnqueueWorkflows';
import { $isReadyToEnqueue } from 'features/queue/store/readiness';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { useCallback } from 'react';
import { serializeError } from 'serialize-error';
import { enqueueMutationFixedCacheKeyOptions, useEnqueueBatchMutation } from 'services/api/endpoints/queue';

const log = logger('generation');

export const useInvoke = () => {
  const dispatch = useAppDispatch();
  const tabName = useAppSelector(selectActiveTab);
  const isReady = useStore($isReadyToEnqueue);
  const isLocked = useIsWorkflowEditorLocked();
  const enqueueWorkflows = useEnqueueWorkflows();

  const [_, { isLoading }] = useEnqueueBatchMutation(enqueueMutationFixedCacheKeyOptions);

  const enqueue = useCallback(
    async (prepend: boolean, isApiValidationRun: boolean) => {
      if (!isReady) {
        return;
      }

      if (tabName === 'workflows') {
        const result = await withResultAsync(() => enqueueWorkflows(prepend, isApiValidationRun));
        if (result.isErr()) {
          log.error({ error: serializeError(result.error) }, 'Failed to enqueue batch');
        } else {
          log.debug(parseify(result.value), 'Enqueued batch');
        }
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
    [dispatch, enqueueWorkflows, isReady, tabName]
  );

  const enqueueBack = useCallback(() => {
    enqueue(false, false);
  }, [enqueue]);

  const enqueueFront = useCallback(() => {
    enqueue(true, false);
  }, [enqueue]);

  return { enqueueBack, enqueueFront, isLoading, isDisabled: !isReady || isLocked, enqueue };
};
