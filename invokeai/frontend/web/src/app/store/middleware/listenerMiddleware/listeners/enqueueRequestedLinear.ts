import { enqueueRequested } from 'app/store/actions';
import { prepareLinearUIBatch } from 'features/nodes/util/graphBuilders/buildLinearBatchConfig';
import { buildLinearImageToImageGraph } from 'features/nodes/util/graphBuilders/buildLinearImageToImageGraph';
import { buildLinearSDXLImageToImageGraph } from 'features/nodes/util/graphBuilders/buildLinearSDXLImageToImageGraph';
import { buildLinearSDXLTextToImageGraph } from 'features/nodes/util/graphBuilders/buildLinearSDXLTextToImageGraph';
import { buildLinearTextToImageGraph } from 'features/nodes/util/graphBuilders/buildLinearTextToImageGraph';
import { queueApi } from 'services/api/endpoints/queue';
import { startAppListening } from '..';

export const addEnqueueRequestedLinear = () => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof enqueueRequested> =>
      enqueueRequested.match(action) &&
      (action.payload.tabName === 'txt2img' ||
        action.payload.tabName === 'img2img'),
    effect: async (action, { getState, dispatch }) => {
      const state = getState();
      const model = state.generation.model;
      const { prepend } = action.payload;

      let graph;

      if (model && model.base_model === 'sdxl') {
        if (action.payload.tabName === 'txt2img') {
          graph = buildLinearSDXLTextToImageGraph(state);
        } else {
          graph = buildLinearSDXLImageToImageGraph(state);
        }
      } else {
        if (action.payload.tabName === 'txt2img') {
          graph = buildLinearTextToImageGraph(state);
        } else {
          graph = buildLinearImageToImageGraph(state);
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
