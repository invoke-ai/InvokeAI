import type { AppStartListening } from 'app/store/store';
import { queueApi, selectQueueStatus } from 'services/api/endpoints/queue';

export const addAnyEnqueuedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: queueApi.endpoints.enqueueBatch.matchFulfilled,
    effect: (_, { dispatch, getState }) => {
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
