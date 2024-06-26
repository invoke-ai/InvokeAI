import { enqueueRequested } from 'app/store/actions';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { getNodeManager } from 'features/controlLayers/konva/nodeManager';
import { stagingAreaBatchIdAdded, stagingAreaInitialized } from 'features/controlLayers/store/canvasV2Slice';
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

      manager.getInpaintMaskImage({ bbox: state.canvasV2.bbox, preview: true });
      manager.getImageSourceImage({ bbox: state.canvasV2.bbox, preview: true });

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
        const enqueueResult = await req.unwrap();
        if (shouldShowProgressInViewer) {
          dispatch(isImageViewerOpenChanged(true));
        }
        // TODO(psyche): update the backend schema, this is always provided
        const batchId = enqueueResult.batch.batch_id;
        assert(batchId, 'No batch ID found in enqueue result');
        if (!state.canvasV2.stagingArea) {
          dispatch(
            stagingAreaInitialized({
              batchIds: [batchId],
              bbox: state.canvasV2.bbox,
            })
          );
        } else {
          dispatch(stagingAreaBatchIdAdded({ batchId }));
        }
      } finally {
        req.reset();
      }
    },
  });
};
