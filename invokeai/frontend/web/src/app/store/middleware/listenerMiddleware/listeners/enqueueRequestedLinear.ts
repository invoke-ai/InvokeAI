import { logger } from 'app/logging/logger';
import { enqueueRequested } from 'app/store/actions';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { $canvasManager } from 'features/controlLayers/konva/CanvasManager';
import { sessionStagingAreaReset, sessionStartedStaging } from 'features/controlLayers/store/canvasV2Slice';
import { prepareLinearUIBatch } from 'features/nodes/util/graph/buildLinearBatchConfig';
import { buildSD1Graph } from 'features/nodes/util/graph/generation/buildSD1Graph';
import { buildSDXLGraph } from 'features/nodes/util/graph/generation/buildSDXLGraph';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { serializeError } from 'serialize-error';
import { queueApi } from 'services/api/endpoints/queue';
import type { Invocation } from 'services/api/types';
import { assert } from 'tsafe';

const log = logger('generation');

export const addEnqueueRequestedLinear = (startAppListening: AppStartListening) => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof enqueueRequested> =>
      enqueueRequested.match(action) && action.payload.tabName === 'generation',
    effect: async (action, { getState, dispatch }) => {
      const state = getState();
      const model = state.canvasV2.params.model;
      const { prepend } = action.payload;

      const manager = $canvasManager.get();
      assert(manager, 'No model found in state');

      let didStartStaging = false;
      if (!state.canvasV2.session.isStaging && state.canvasV2.session.mode === 'compose') {
        dispatch(sessionStartedStaging());
        didStartStaging = true;
      }

      try {
        let g: Graph;
        let noise: Invocation<'noise'>;
        let posCond: Invocation<'compel' | 'sdxl_compel_prompt'>;

        assert(model, 'No model found in state');
        const base = model.base;

        if (base === 'sdxl') {
          const result = await buildSDXLGraph(state, manager);
          g = result.g;
          noise = result.noise;
          posCond = result.posCond;
        } else if (base === 'sd-1' || base === 'sd-2') {
          const result = await buildSD1Graph(state, manager);
          g = result.g;
          noise = result.noise;
          posCond = result.posCond;
        } else {
          assert(false, `No graph builders for base ${base}`);
        }

        const batchConfig = prepareLinearUIBatch(state, g, prepend, noise, posCond);

        const req = dispatch(
          queueApi.endpoints.enqueueBatch.initiate(batchConfig, {
            fixedCacheKey: 'enqueueBatch',
          })
        );
        req.reset();
        await req.unwrap();
      } catch (error) {
        log.error({ error: serializeError(error) }, 'Failed to enqueue batch');
        if (didStartStaging && getState().canvasV2.session.isStaging) {
          dispatch(sessionStagingAreaReset());
        }
      }
    },
  });
};
