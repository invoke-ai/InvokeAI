import { isAnyOf } from '@reduxjs/toolkit';
import { queueApi } from 'services/api/endpoints/queue';
import { startAppListening } from '..';

const matcher = isAnyOf(
  queueApi.endpoints.enqueueBatch.matchFulfilled,
  queueApi.endpoints.enqueueGraph.matchFulfilled
);

export const addAnyEnqueuedListener = () => {
  startAppListening({
    matcher,
    effect: async (_, { dispatch, getState }) => {
      const { data } = queueApi.endpoints.getQueueStatus.select()(getState());

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
