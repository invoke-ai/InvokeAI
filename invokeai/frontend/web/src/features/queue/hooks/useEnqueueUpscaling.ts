import { createAction } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { useAppStore } from 'app/store/storeHooks';
import { positivePromptAddedToHistory, selectPositivePrompt } from 'features/controlLayers/store/paramsSlice';
import { prepareLinearUIBatch } from 'features/nodes/util/graph/buildLinearBatchConfig';
import { buildMultidiffusionUpscaleGraph } from 'features/nodes/util/graph/buildMultidiffusionUpscaleGraph';
import { useCallback } from 'react';

import type { EnqueueBatchArg } from './utils/executeEnqueue';
import { executeEnqueue } from './utils/executeEnqueue';

export const enqueueRequestedUpscaling = createAction('app/enqueueRequestedUpscaling');

const log = logger('generation');

type UpscaleBuildResult = {
  batchConfig: EnqueueBatchArg;
};

export const useEnqueueUpscaling = () => {
  const store = useAppStore();

  const enqueue = useCallback(
    (prepend: boolean) => {
      return executeEnqueue({
        store,
        options: { prepend },
        requestedAction: enqueueRequestedUpscaling,
        log,
        build: async ({ store: innerStore, options }) => {
          const state = innerStore.getState();
          const model = state.params.model;
          if (!model) {
            log.error('No model found in state');
            return null;
          }

          const { g, seed, positivePrompt } = await buildMultidiffusionUpscaleGraph(state);

          const batchConfig = prepareLinearUIBatch({
            state,
            g,
            base: model.base,
            prepend: options.prepend,
            seedNode: seed,
            positivePromptNode: positivePrompt,
            origin: 'upscaling',
            destination: 'gallery',
          });

          return { batchConfig } satisfies UpscaleBuildResult;
        },
        prepareBatch: ({ buildResult }) => buildResult.batchConfig,
        onSuccess: ({ store: innerStore }) => {
          const state = innerStore.getState();
          innerStore.dispatch(positivePromptAddedToHistory(selectPositivePrompt(state)));
        },
      });
    },
    [store]
  );

  return enqueue;
};
