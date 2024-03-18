import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { queueApi, selectQueueStatus } from 'services/api/endpoints/queue';

export const addAnyEnqueuedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: queueApi.endpoints.enqueueBatch.matchFulfilled,
    effect: async (_, { dispatch, getState }) => {
      const { data } = selectQueueStatus(getState());

      if (!data || data.processor.is_started) {
        return;
      }

      dispatch(
        queueApi.endpoints.resumeProcessor.initiate(undefined, {
          fixedCacheKey: 'resumeProcessor',
        })
      );
    },
  });
};
