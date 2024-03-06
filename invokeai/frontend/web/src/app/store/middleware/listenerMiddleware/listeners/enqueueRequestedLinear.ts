import { enqueueRequested } from 'app/store/actions';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { prepareLinearUIBatch } from 'features/nodes/util/graph/buildLinearBatchConfig';
import { buildLinearImageToImageGraph } from 'features/nodes/util/graph/buildLinearImageToImageGraph';
import { buildLinearSDXLImageToImageGraph } from 'features/nodes/util/graph/buildLinearSDXLImageToImageGraph';
import { buildLinearSDXLTextToImageGraph } from 'features/nodes/util/graph/buildLinearSDXLTextToImageGraph';
import { buildLinearTextToImageGraph } from 'features/nodes/util/graph/buildLinearTextToImageGraph';
import { queueApi } from 'services/api/endpoints/queue';

export const addEnqueueRequestedLinear = (startAppListening: AppStartListening) => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof enqueueRequested> =>
      enqueueRequested.match(action) && (action.payload.tabName === 'txt2img' || action.payload.tabName === 'img2img'),
    effect: async (action, { getState, dispatch }) => {
      const state = getState();
      const model = state.generation.model;
      const { prepend } = action.payload;

      let graph;

      if (model && model.base === 'sdxl') {
        if (action.payload.tabName === 'txt2img') {
          graph = await buildLinearSDXLTextToImageGraph(state);
        } else {
          graph = await buildLinearSDXLImageToImageGraph(state);
        }
      } else {
        if (action.payload.tabName === 'txt2img') {
          graph = await buildLinearTextToImageGraph(state);
        } else {
          graph = await buildLinearImageToImageGraph(state);
        }
      }

      const batchConfig = prepareLinearUIBatch(state, graph, prepend);

      const req = dispatch(
        queueApi.endpoints.enqueueBatch.initiate(batchConfig, {
          fixedCacheKey: 'enqueueBatch',
        })
      );
      req.reset();
    },
  });
};
