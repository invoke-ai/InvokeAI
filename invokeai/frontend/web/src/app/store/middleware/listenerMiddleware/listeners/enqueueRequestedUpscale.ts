import { logger } from 'app/logging/logger';
import { enqueueRequested } from 'app/store/actions';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { parseify } from 'common/util/serialize';
import { prepareLinearUIBatch } from 'features/nodes/util/graph/buildLinearBatchConfig';
import { buildMultidiffusionUpscaleGraph } from 'features/nodes/util/graph/buildMultidiffusionUpscaleGraph';
import { serializeError } from 'serialize-error';
import { enqueueMutationFixedCacheKeyOptions, queueApi } from 'services/api/endpoints/queue';

const log = logger('generation');

export const addEnqueueRequestedUpscale = (startAppListening: AppStartListening) => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof enqueueRequested> =>
      enqueueRequested.match(action) && action.payload.tabName === 'upscaling',
    effect: async (action, { getState, dispatch }) => {
      const state = getState();
      const { prepend } = action.payload;

      const { g, noise, posCond } = await buildMultidiffusionUpscaleGraph(state);

      const batchConfig = prepareLinearUIBatch(state, g, prepend, noise, posCond, 'upscaling', 'gallery');

      const req = dispatch(queueApi.endpoints.enqueueBatch.initiate(batchConfig, enqueueMutationFixedCacheKeyOptions));
      try {
        await req.unwrap();
        log.debug(parseify({ batchConfig }), 'Enqueued batch');
      } catch (error) {
        log.error({ error: serializeError(error) }, 'Failed to enqueue batch');
      } finally {
        req.reset();
      }
    },
  });
};
