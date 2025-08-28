import { createAction } from '@reduxjs/toolkit';
import type { AppStore } from 'app/store/store';
import { useAppStore } from 'app/store/storeHooks';
import { prepareLinearUIBatch } from 'features/nodes/util/graph/buildLinearBatchConfig';
import { buildMultidiffusionUpscaleGraph } from 'features/nodes/util/graph/buildMultidiffusionUpscaleGraph';
import { useCallback } from 'react';
import { enqueueMutationFixedCacheKeyOptions, queueApi } from 'services/api/endpoints/queue';

export const enqueueRequestedUpscaling = createAction('app/enqueueRequestedUpscaling');

const enqueueUpscaling = async (store: AppStore, prepend: boolean) => {
  const { dispatch, getState } = store;

  dispatch(enqueueRequestedUpscaling());

  const state = getState();

  const { g, seed, positivePrompt } = await buildMultidiffusionUpscaleGraph(state);

  const batchConfig = prepareLinearUIBatch({
    state,
    g,
    prepend,
    seedNode: seed,
    positivePromptNode: positivePrompt,
    origin: 'upscaling',
    destination: 'gallery',
  });

  const req = dispatch(
    queueApi.endpoints.enqueueBatch.initiate(batchConfig, { ...enqueueMutationFixedCacheKeyOptions, track: false })
  );
  const enqueueResult = await req.unwrap();

  return { batchConfig, enqueueResult };
};

export const useEnqueueUpscaling = () => {
  const store = useAppStore();
  const enqueue = useCallback(
    (prepend: boolean) => {
      return enqueueUpscaling(store, prepend);
    },
    [store]
  );
  return enqueue;
};
