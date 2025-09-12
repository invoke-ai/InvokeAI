import type { AlertStatus } from '@invoke-ai/ui-library';
import { createAction } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { AppStore } from 'app/store/store';
import { useAppStore } from 'app/store/storeHooks';
import { extractMessageFromAssertionError } from 'common/util/extractMessageFromAssertionError';
import { withResult, withResultAsync } from 'common/util/result';
import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { positivePromptAddedToHistory, selectPositivePrompt } from 'features/controlLayers/store/paramsSlice';
import { prepareLinearUIBatch } from 'features/nodes/util/graph/buildLinearBatchConfig';
import { buildChatGPT4oGraph } from 'features/nodes/util/graph/generation/buildChatGPT4oGraph';
import { buildCogView4Graph } from 'features/nodes/util/graph/generation/buildCogView4Graph';
import { buildFLUXGraph } from 'features/nodes/util/graph/generation/buildFLUXGraph';
import { buildFluxKontextGraph } from 'features/nodes/util/graph/generation/buildFluxKontextGraph';
import { buildGemini2_5Graph } from 'features/nodes/util/graph/generation/buildGemini2_5Graph';
import { buildImagen3Graph } from 'features/nodes/util/graph/generation/buildImagen3Graph';
import { buildImagen4Graph } from 'features/nodes/util/graph/generation/buildImagen4Graph';
import { buildSD1Graph } from 'features/nodes/util/graph/generation/buildSD1Graph';
import { buildSD3Graph } from 'features/nodes/util/graph/generation/buildSD3Graph';
import { buildSDXLGraph } from 'features/nodes/util/graph/generation/buildSDXLGraph';
import { selectCanvasDestination } from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg } from 'features/nodes/util/graph/types';
import { UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { serializeError } from 'serialize-error';
import { enqueueMutationFixedCacheKeyOptions, queueApi } from 'services/api/endpoints/queue';
import { assert, AssertionError } from 'tsafe';

const log = logger('generation');
export const enqueueRequestedCanvas = createAction('app/enqueueRequestedCanvas');

const enqueueCanvas = async (store: AppStore, canvasManager: CanvasManager, prepend: boolean) => {
  const { dispatch, getState } = store;

  dispatch(enqueueRequestedCanvas());

  const state = getState();

  const destination = selectCanvasDestination(state);

  const model = state.params.model;
  if (!model) {
    log.error('No model found in state');
    return;
  }

  const base = model.base;

  const buildGraphResult = await withResultAsync(async () => {
    const generationMode = await canvasManager.compositor.getGenerationMode();
    const graphBuilderArg: GraphBuilderArg = { generationMode, state, manager: canvasManager };

    switch (base) {
      case 'sdxl':
        return await buildSDXLGraph(graphBuilderArg);
      case 'sd-1':
      case `sd-2`:
        return await buildSD1Graph(graphBuilderArg);
      case `sd-3`:
        return await buildSD3Graph(graphBuilderArg);
      case `flux`:
        return await buildFLUXGraph(graphBuilderArg);
      case 'cogview4':
        return await buildCogView4Graph(graphBuilderArg);
      case 'imagen3':
        return buildImagen3Graph(graphBuilderArg);
      case 'imagen4':
        return buildImagen4Graph(graphBuilderArg);
      case 'chatgpt-4o':
        return await buildChatGPT4oGraph(graphBuilderArg);
      case 'flux-kontext':
        return buildFluxKontextGraph(graphBuilderArg);
      case 'gemini-2.5':
        return buildGemini2_5Graph(graphBuilderArg);
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

  const { g, seed, positivePrompt } = buildGraphResult.value;

  const prepareBatchResult = withResult(() =>
    prepareLinearUIBatch({
      state,
      g,
      base,
      prepend,
      seedNode: seed,
      positivePromptNode: positivePrompt,
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

  // Push to prompt history on successful enqueue
  dispatch(positivePromptAddedToHistory(selectPositivePrompt(state)));

  return { batchConfig, enqueueResult };
};

export const useEnqueueCanvas = () => {
  const store = useAppStore();
  const canvasManager = useCanvasManagerSafe();
  const enqueue = useCallback(
    (prepend: boolean) => {
      if (!canvasManager) {
        log.error('Canvas manager is not available');
        return;
      }
      return enqueueCanvas(store, canvasManager, prepend);
    },
    [canvasManager, store]
  );
  return enqueue;
};
