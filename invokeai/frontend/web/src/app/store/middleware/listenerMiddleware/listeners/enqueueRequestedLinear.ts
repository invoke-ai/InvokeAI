import { enqueueRequested } from 'app/store/actions';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { getNodeManager } from 'features/controlLayers/konva/KonvaNodeManager';
import { stagingAreaCanceledStaging, stagingAreaStartedStaging } from 'features/controlLayers/store/canvasV2Slice';
import { prepareLinearUIBatch } from 'features/nodes/util/graph/buildLinearBatchConfig';
import { buildSD1Graph } from 'features/nodes/util/graph/generation/buildSD1Graph';
import { buildSDXLGraph } from 'features/nodes/util/graph/generation/buildSDXLGraph';
import { queueApi } from 'services/api/endpoints/queue';
import { assert } from 'tsafe';

export const addEnqueueRequestedLinear = (startAppListening: AppStartListening) => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof enqueueRequested> =>
      enqueueRequested.match(action) && action.payload.tabName === 'generation',
    effect: async (action, { getState, dispatch }) => {
      const state = getState();
      const model = state.canvasV2.params.model;
      const { prepend } = action.payload;

      let didStartStaging = false;
      if (!state.canvasV2.stagingArea.isStaging) {
        dispatch(stagingAreaStartedStaging());
        didStartStaging = true;
      }

      try {
        let g;

        const manager = getNodeManager();
        assert(model, 'No model found in state');
        const base = model.base;

        if (base === 'sdxl') {
          g = await buildSDXLGraph(state, manager);
        } else if (base === 'sd-1' || base === 'sd-2') {
          g = await buildSD1Graph(state, manager);
        } else {
          assert(false, `No graph builders for base ${base}`);
        }

        const batchConfig = prepareLinearUIBatch(state, g, prepend);

        const req = dispatch(
          queueApi.endpoints.enqueueBatch.initiate(batchConfig, {
            fixedCacheKey: 'enqueueBatch',
          })
        );
        req.reset();
        await req.unwrap();
      } catch {
        if (didStartStaging && getState().canvasV2.stagingArea.isStaging) {
          dispatch(stagingAreaCanceledStaging());
        }
      }
    },
  });
};
