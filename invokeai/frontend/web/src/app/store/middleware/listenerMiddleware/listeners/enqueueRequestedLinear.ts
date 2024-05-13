import { enqueueRequested } from 'app/store/actions';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { isImageViewerOpenChanged } from 'features/gallery/store/gallerySlice';
import { buildGenerationTabGraph } from 'features/nodes/util/graph/buildGenerationTabGraph';
import { buildGenerationTabSDXLGraph } from 'features/nodes/util/graph/buildGenerationTabSDXLGraph';
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

      if (model?.base === 'sdxl') {
        graph = await buildGenerationTabSDXLGraph(state);
      } else {
        graph = await buildGenerationTabGraph(state);
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
