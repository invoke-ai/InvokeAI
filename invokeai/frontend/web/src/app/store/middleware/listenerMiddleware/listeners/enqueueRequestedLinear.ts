import { enqueueRequested } from 'app/store/actions';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { getNodeManager } from 'features/controlLayers/konva/nodeManager';
import { isImageViewerOpenChanged } from 'features/gallery/store/gallerySlice';
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
      const { shouldShowProgressInViewer } = state.ui;
      const model = state.canvasV2.params.model;
      const { prepend } = action.payload;

      let graph;

      const manager = getNodeManager();
      assert(model, 'No model found in state');
      const base = model.base;

      if (base === 'sdxl') {
        graph = await buildSDXLGraph(state, manager);
      } else if (base === 'sd-1' || base === 'sd-2') {
        graph = await buildSD1Graph(state, manager);
      } else {
        assert(false, `No graph builders for base ${base}`);
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
