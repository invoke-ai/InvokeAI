import type { AlertStatus } from '@invoke-ai/ui-library';
import { createAction } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { AppStore } from 'app/store/store';
import { useAppStore } from 'app/store/storeHooks';
import { extractMessageFromAssertionError } from 'common/util/extractMessageFromAssertionError';
import { withResult, withResultAsync } from 'common/util/result';
import { generateSessionIdCreated, selectGenerateSessionId } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { prepareLinearUIBatch } from 'features/nodes/util/graph/buildLinearBatchConfig';
import { buildChatGPT4oGraph } from 'features/nodes/util/graph/generation/buildChatGPT4oGraph';
import { buildCogView4Graph } from 'features/nodes/util/graph/generation/buildCogView4Graph';
import { buildFLUXGraph } from 'features/nodes/util/graph/generation/buildFLUXGraph';
import { buildFluxKontextGraph } from 'features/nodes/util/graph/generation/buildFluxKontextGraph';
import { buildImagen3Graph } from 'features/nodes/util/graph/generation/buildImagen3Graph';
import { buildImagen4Graph } from 'features/nodes/util/graph/generation/buildImagen4Graph';
import { buildSD1Graph } from 'features/nodes/util/graph/generation/buildSD1Graph';
import { buildSD3Graph } from 'features/nodes/util/graph/generation/buildSD3Graph';
import { buildSDXLGraph } from 'features/nodes/util/graph/generation/buildSDXLGraph';
import { UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { serializeError } from 'serialize-error';
import { enqueueMutationFixedCacheKeyOptions, queueApi } from 'services/api/endpoints/queue';
import { assert, AssertionError } from 'tsafe';

const log = logger('generation');

export const enqueueRequestedGenerate = createAction('app/enqueueRequestedGenerate');

const enqueueGenerate = async (store: AppStore, prepend: boolean) => {
  const { dispatch, getState } = store;

  dispatch(enqueueRequestedGenerate());

  let destination = selectGenerateSessionId(getState());
  if (!destination) {
    dispatch(generateSessionIdCreated());
    destination = selectGenerateSessionId(getState());
  }
  assert(destination !== null);

  const state = getState();

  const model = state.params.model;
  assert(model, 'No model found in state');
  const base = model.base;

  const buildGraphResult = await withResultAsync(async () => {
    switch (base) {
      case 'sdxl':
        return await buildSDXLGraph(state, null);
      case 'sd-1':
      case `sd-2`:
        return await buildSD1Graph(state, null);
      case `sd-3`:
        return await buildSD3Graph(state, null);
      case `flux`:
        return await buildFLUXGraph(state, null);
      case 'cogview4':
        return await buildCogView4Graph(state, null);
      case 'imagen3':
        return await buildImagen3Graph(state, null);
      case 'imagen4':
        return await buildImagen4Graph(state, null);
      case 'chatgpt-4o':
        return await buildChatGPT4oGraph(state, null);
      case 'flux-kontext':
        return await buildFluxKontextGraph(state, null);
      default:
        assert(false, `No graph builders for base ${base}`);
    }
  });

  if (buildGraphResult.isErr()) {
    let title = 'Failed to build graph';
    let status: AlertStatus = 'error';
    let description: string | null = null;
    if (buildGraphResult.error instanceof AssertionError) {
      description = extractMessageFromAssertionError(buildGraphResult.error);
    } else if (buildGraphResult.error instanceof UnsupportedGenerationModeError) {
      title = 'Unsupported generation mode';
      description = buildGraphResult.error.message;
      status = 'warning';
    }
    const error = serializeError(buildGraphResult.error);
    log.error({ error }, 'Failed to build graph');
    toast({
      status,
      title,
      description,
    });
    return;
  }

  const { g, seedFieldIdentifier, positivePromptFieldIdentifier } = buildGraphResult.value;

  const prepareBatchResult = withResult(() =>
    prepareLinearUIBatch({
      state,
      g,
      prepend,
      seedFieldIdentifier,
      positivePromptFieldIdentifier,
      origin: 'canvas',
      destination,
    })
  );

  if (prepareBatchResult.isErr()) {
    log.error({ error: serializeError(prepareBatchResult.error) }, 'Failed to prepare batch');
    return;
  }

  const batchConfig = prepareBatchResult.value;

  const req = dispatch(
    queueApi.endpoints.enqueueBatch.initiate(batchConfig, {
      ...enqueueMutationFixedCacheKeyOptions,
      track: false,
    })
  );

  const enqueueResult = await req.unwrap();

  return { batchConfig, enqueueResult };
};

export const useEnqueueGenerate = () => {
  const store = useAppStore();
  const enqueue = useCallback(
    (prepend: boolean) => {
      return enqueueGenerate(store, prepend);
    },
    [store]
  );
  return enqueue;
};
