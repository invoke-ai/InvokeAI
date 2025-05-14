import { createAction } from '@reduxjs/toolkit';
import { useAppStore } from 'app/store/nanostores/store';
import { buildSimpleGraph } from 'features/nodes/util/graph/buildSimpleGraph';
import { useCallback } from 'react';
import { enqueueMutationFixedCacheKeyOptions, queueApi } from 'services/api/endpoints/queue';
import type { EnqueueBatchArg } from 'services/api/types';

const enqueueRequestedSimple = createAction('app/enqueueRequestedSimple');

export const useEnqueueSimple = () => {
  const { getState, dispatch } = useAppStore();
  const enqueue = useCallback(
    async (prepend: boolean) => {
      dispatch(enqueueRequestedSimple());
      const state = getState();
      const g = buildSimpleGraph(state);

      const batchConfig: EnqueueBatchArg = {
        batch: {
          graph: g.getGraph(),
          runs: state.params.iterations,
          origin: 'simple',
          destination: 'gallery',
        },
        prepend,
      };

      const req = dispatch(
        queueApi.endpoints.enqueueBatch.initiate(batchConfig, { ...enqueueMutationFixedCacheKeyOptions, track: false })
      );

      const enqueueResult = await req.unwrap();
      return { batchConfig, enqueueResult };
    },
    [dispatch, getState]
  );

  return enqueue;
};
