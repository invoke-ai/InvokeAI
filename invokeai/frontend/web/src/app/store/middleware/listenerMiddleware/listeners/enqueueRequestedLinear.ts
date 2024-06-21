import { enqueueRequested } from 'app/store/actions';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { $nodeManager } from 'features/controlLayers/konva/renderers/renderer';
import { isImageViewerOpenChanged } from 'features/gallery/store/gallerySlice';
import { prepareLinearUIBatch } from 'features/nodes/util/graph/buildLinearBatchConfig';
import { buildGenerationTabGraph } from 'features/nodes/util/graph/generation/buildGenerationTabGraph';
import { buildGenerationTabSDXLGraph } from 'features/nodes/util/graph/generation/buildGenerationTabSDXLGraph';
import { queueApi } from 'services/api/endpoints/queue';
import { assert } from 'tsafe';

export const addEnqueueRequestedLinear = (startAppListening: AppStartListening) => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof enqueueRequested> =>
      enqueueRequested.match(action) && action.payload.tabName === 'generation',
    effect: async (action, { getState, dispatch }) => {
      const state = getState();
      const { shouldShowProgressInViewer } = state.ui;
      const model = state.canvasV2.params.model;
      const { prepend } = action.payload;

      let graph;

      const manager = $nodeManager.get();
      assert(manager, 'Konva node manager not initialized');

      console.log('generation mode', manager.util.getGenerationMode());

      if (model?.base === 'sdxl') {
        graph = await buildGenerationTabSDXLGraph(state, manager);
      } else {
        graph = await buildGenerationTabGraph(state, manager);
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
