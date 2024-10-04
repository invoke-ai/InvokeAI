import { enqueueRequested } from 'app/store/actions';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { prepareLinearUIBatch } from 'features/nodes/util/graph/buildLinearBatchConfig';
import { buildMultidiffusionUpscaleGraph } from 'features/nodes/util/graph/buildMultidiffusionUpscaleGraph';
import { queueApi } from 'services/api/endpoints/queue';

export const addEnqueueRequestedUpscale = (startAppListening: AppStartListening) => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof enqueueRequested> =>
      enqueueRequested.match(action) && action.payload.tabName === 'upscaling',
    effect: async (action, { getState, dispatch }) => {
      const state = getState();
      const { prepend } = action.payload;

      const { g, noise, posCond } = await buildMultidiffusionUpscaleGraph(state);

      const batchConfig = prepareLinearUIBatch(state, g, prepend, noise, posCond, 'upscaling', 'gallery');

      const req = dispatch(
        queueApi.endpoints.enqueueBatch.initiate(batchConfig, {
          fixedCacheKey: 'enqueueBatch',
        })
      );
      try {
        await req.unwrap();
      } finally {
        req.reset();
      }
    },
  });
};
