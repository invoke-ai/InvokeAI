import { createAction } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { extractMessageFromAssertionError } from 'common/util/extractMessageFromAssertionError';
import { withResult, withResultAsync } from 'common/util/result';
import { parseify } from 'common/util/serialize';
import { $canvasManager } from 'features/controlLayers/store/ephemeral';
import { prepareLinearUIBatch } from 'features/nodes/util/graph/buildLinearBatchConfig';
import { buildFLUXGraph } from 'features/nodes/util/graph/generation/buildFLUXGraph';
import { buildSD1Graph } from 'features/nodes/util/graph/generation/buildSD1Graph';
import { buildSD3Graph } from 'features/nodes/util/graph/generation/buildSD3Graph';
import { buildSDXLGraph } from 'features/nodes/util/graph/generation/buildSDXLGraph';
import { toast } from 'features/toast/toast';
import { serializeError } from 'serialize-error';
import { enqueueMutationFixedCacheKeyOptions, queueApi } from 'services/api/endpoints/queue';
import { assert, AssertionError } from 'tsafe';

const log = logger('generation');

export const enqueueRequestedCanvas = createAction<{ prepend: boolean }>('app/enqueueRequestedCanvas');

export const addEnqueueRequestedLinear = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: enqueueRequestedCanvas,
    effect: async (action, { getState, dispatch }) => {
      log.debug('Enqueue requested');
      const state = getState();
      const { prepend } = action.payload;

      const manager = $canvasManager.get();
      assert(manager, 'No canvas manager');

      const model = state.params.model;
      assert(model, 'No model found in state');
      const base = model.base;

      const buildGraphResult = await withResultAsync(async () => {
        switch (base) {
          case 'sdxl':
            return await buildSDXLGraph(state, manager);
          case 'sd-1':
          case `sd-2`:
            return await buildSD1Graph(state, manager);
          case `sd-3`:
            return await buildSD3Graph(state, manager);
          case `flux`:
            return await buildFLUXGraph(state, manager);
          default:
            assert(false, `No graph builders for base ${base}`);
        }
      });

      if (buildGraphResult.isErr()) {
        let description: string | null = null;
        if (buildGraphResult.error instanceof AssertionError) {
          description = extractMessageFromAssertionError(buildGraphResult.error);
        }
        const error = serializeError(buildGraphResult.error);
        log.error({ error }, 'Failed to build graph');
        toast({
          status: 'error',
          title: 'Failed to build graph',
          description,
        });
        return;
      }

      const { g, noise, posCond } = buildGraphResult.value;

      const destination = state.canvasSettings.sendToCanvas ? 'canvas' : 'gallery';

      const prepareBatchResult = withResult(() =>
        prepareLinearUIBatch(state, g, prepend, noise, posCond, 'canvas', destination)
      );

      if (prepareBatchResult.isErr()) {
        log.error({ error: serializeError(prepareBatchResult.error) }, 'Failed to prepare batch');
        return;
      }

      const req = dispatch(
        queueApi.endpoints.enqueueBatch.initiate(prepareBatchResult.value, enqueueMutationFixedCacheKeyOptions)
      );

      try {
        await req.unwrap();
        log.debug(parseify({ batchConfig: prepareBatchResult.value }), 'Enqueued batch');
      } catch (error) {
        log.error({ error: serializeError(error) }, 'Failed to enqueue batch');
      } finally {
        req.reset();
      }
    },
  });
};
