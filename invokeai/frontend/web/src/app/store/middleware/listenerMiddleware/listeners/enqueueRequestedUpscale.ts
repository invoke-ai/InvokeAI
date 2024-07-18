import { enqueueRequested } from 'app/store/actions';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { isImageViewerOpenChanged } from 'features/gallery/store/gallerySlice';
import { prepareLinearUIBatch } from 'features/nodes/util/graph/buildLinearBatchConfig';
import { buildMultidiffusionUpscsaleGraph } from 'features/nodes/util/graph/buildMultidiffusionUpscaleGraph';
import { queueApi } from 'services/api/endpoints/queue';

export const addEnqueueRequestedUpscale = (startAppListening: AppStartListening) => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof enqueueRequested> =>
      enqueueRequested.match(action) && action.payload.tabName === 'upscaling',
    effect: async (action, { getState, dispatch }) => {

      const state = getState();
      const { shouldShowProgressInViewer } = state.ui;
      const { prepend } = action.payload;


      const graph = await buildMultidiffusionUpscsaleGraph(state)

      const batchConfig = prepareLinearUIBatch(state, graph, prepend);

      const req = dispatch(
        queueApi.endpoints.enqueueBatch.initiate(batchConfig, {
          fixedCacheKey: 'enqueueBatch',
        })
      );
      try {
        await req.unwrap();
        if (shouldShowProgressInViewer) {
          dispatch(isImageViewerOpenChanged(true));
        }
      } finally {
        req.reset();
      }
    },
  });
};
