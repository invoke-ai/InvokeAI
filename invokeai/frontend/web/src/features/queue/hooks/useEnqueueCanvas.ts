import type { AlertStatus } from '@invoke-ai/ui-library';
import { createAction } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { useAppStore } from 'app/store/storeHooks';
import { extractMessageFromAssertionError } from 'common/util/extractMessageFromAssertionError';
import { withResult, withResultAsync } from 'common/util/result';
import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { positivePromptAddedToHistory, selectPositivePrompt } from 'features/controlLayers/store/paramsSlice';
import { prepareLinearUIBatch } from 'features/nodes/util/graph/buildLinearBatchConfig';
import { selectCanvasDestination } from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg } from 'features/nodes/util/graph/types';
import { UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { serializeError } from 'serialize-error';
import { AssertionError } from 'tsafe';

import type { EnqueueBatchArg } from './utils/executeEnqueue';
import { executeEnqueue } from './utils/executeEnqueue';
import { buildGraphForBase } from './utils/graphBuilders';

const log = logger('generation');
export const enqueueRequestedCanvas = createAction('app/enqueueRequestedCanvas');

type CanvasBuildResult = {
  batchConfig: EnqueueBatchArg;
};

export const useEnqueueCanvas = () => {
  const store = useAppStore();
  const canvasManager = useCanvasManagerSafe();

  const enqueue = useCallback(
    (prepend: boolean) => {
      if (!canvasManager) {
        log.error('Canvas manager is not available');
        return null;
      }

      return executeEnqueue({
        store,
        options: { prepend },
        requestedAction: enqueueRequestedCanvas,
        log,
        build: async ({ store: innerStore, options }) => {
          const state = innerStore.getState();

          const destination = selectCanvasDestination(state);
          const model = state.params.model;
          if (!model) {
            log.error('No model found in state');
            return null;
          }

          const generationMode = await canvasManager.compositor.getGenerationMode();
          const graphBuilderArg: GraphBuilderArg = { generationMode, state, manager: canvasManager };

          const buildGraphResult = await withResultAsync(
            async () => await buildGraphForBase(model.base, graphBuilderArg)
          );

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
            toast({ status, title, description });
            return null;
          }

          const { g, seed, positivePrompt } = buildGraphResult.value;

          const prepareBatchResult = withResult(() =>
            prepareLinearUIBatch({
              state,
              g,
              base: model.base,
              prepend: options.prepend,
              seedNode: seed,
              positivePromptNode: positivePrompt,
              origin: 'canvas',
              destination,
            })
          );

          if (prepareBatchResult.isErr()) {
            log.error({ error: serializeError(prepareBatchResult.error) }, 'Failed to prepare batch');
            return null;
          }

          return {
            batchConfig: prepareBatchResult.value,
          } satisfies CanvasBuildResult;
        },
        prepareBatch: ({ buildResult }) => buildResult.batchConfig,
        onSuccess: ({ store: innerStore }) => {
          const state = innerStore.getState();
          innerStore.dispatch(positivePromptAddedToHistory(selectPositivePrompt(state)));
        },
      });
    },
    [canvasManager, store]
  );

  return enqueue;
};
