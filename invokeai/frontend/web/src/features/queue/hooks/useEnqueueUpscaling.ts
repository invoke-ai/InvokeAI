import { logger } from 'app/logging/logger';
import type { AppStore } from 'app/store/store';
import { useAppStore } from 'app/store/storeHooks';
import { positivePromptAddedToHistory, selectPositivePrompt } from 'features/controlLayers/store/paramsSlice';
import { buildFluxMultidiffusionUpscaleGraph } from 'features/nodes/util/graph/buildFluxMultidiffusionUpscaleGraph';
import type { BaseModelType } from 'features/nodes/types/common';
import { prepareLinearUIBatch } from 'features/nodes/util/graph/buildLinearBatchConfig';
import { buildMultidiffusionUpscaleGraph } from 'features/nodes/util/graph/buildMultidiffusionUpscaleGraph';
import { buildZImageMultidiffusionUpscaleGraph } from 'features/nodes/util/graph/buildZImageMultidiffusionUpscaleGraph';
import { useCallback } from 'react';
import { enqueueMutationFixedCacheKeyOptions, queueApi } from 'services/api/endpoints/queue';

const log = logger('generation');

const buildUpscaleGraph = (state: ReturnType<AppStore['getState']>, base: string) => {
  switch (base) {
    case 'flux':
      return buildFluxMultidiffusionUpscaleGraph(state);
    case 'z-image':
      return buildZImageMultidiffusionUpscaleGraph(state);
    default:
      return buildMultidiffusionUpscaleGraph(state);
  }
};

const enqueueUpscaling = async (store: AppStore, prepend: boolean) => {
  const { dispatch, getState } = store;

  const state = getState();

  const model = state.params.model;
  if (!model) {
    log.error('No model found in state');
    return;
  }
  const base = model.base;

  const { g, seed, positivePrompt } = await buildUpscaleGraph(state, base);

  const batchConfig = prepareLinearUIBatch({
    state,
    g,
    base: base as BaseModelType,
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

  // Push to prompt history on successful enqueue
  dispatch(positivePromptAddedToHistory(selectPositivePrompt(state)));

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
