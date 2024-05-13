import { enqueueRequested } from 'app/store/actions';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { isImageViewerOpenChanged } from 'features/gallery/store/gallerySlice';
import { buildGenerationTabGraph2 } from 'features/nodes/util/graph/buildGenerationTabGraph2';
import { buildGenerationTabSDXLGraph2 } from 'features/nodes/util/graph/buildGenerationTabSDXLGraph2';
import { prepareLinearUIBatch } from 'features/nodes/util/graph/buildLinearBatchConfig';
import { queueApi } from 'services/api/endpoints/queue';

export const addEnqueueRequestedLinear = (startAppListening: AppStartListening) => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof enqueueRequested> =>
      enqueueRequested.match(action) && action.payload.tabName === 'generation',
    effect: async (action, { getState, dispatch }) => {
      const state = getState();
      const { shouldShowProgressInViewer } = state.ui;
      const model = state.generation.model;
      const { prepend } = action.payload;

      let graph;

      if (model && model.base === 'sdxl') {
        graph = await buildGenerationTabSDXLGraph2(state);
      } else {
        graph = await buildGenerationTabGraph2(state);
      }

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
