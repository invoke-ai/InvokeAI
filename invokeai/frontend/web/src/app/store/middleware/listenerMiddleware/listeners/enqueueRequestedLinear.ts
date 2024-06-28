import { enqueueRequested } from 'app/store/actions';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { getNodeManager } from 'features/controlLayers/konva/nodeManager';
import {
  stagingAreaBatchIdAdded,
  stagingAreaInitialized,
  stagingAreaReset,
} from 'features/controlLayers/store/canvasV2Slice';
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

      let didInitializeStagingArea = false;

      if (state.canvasV2.stagingArea === null) {
        dispatch(
          stagingAreaInitialized({
            batchIds: [],
            bbox: state.canvasV2.bbox,
          })
        );
        didInitializeStagingArea = true;
      }

      try {
        let g;

        const manager = getNodeManager();
        assert(model, 'No model found in state');
        const base = model.base;

        if (base === 'sdxl') {
          g = await buildSDXLGraph(state, manager);
        } else if (base === 'sd-1' || base === 'sd-2') {
          g = await buildSD1Graph(state, manager);
        } else {
          assert(false, `No graph builders for base ${base}`);
        }

        const batchConfig = prepareLinearUIBatch(state, g, prepend);

        const req = dispatch(
          queueApi.endpoints.enqueueBatch.initiate(batchConfig, {
            fixedCacheKey: 'enqueueBatch',
          })
        );

        const enqueueResult = await req.unwrap();
        req.reset();

        if (shouldShowProgressInViewer) {
          dispatch(isImageViewerOpenChanged(true));
        }
        // TODO(psyche): update the backend schema, this is always provided
        const batchId = enqueueResult.batch.batch_id;
        assert(batchId, 'No batch ID found in enqueue result');
        dispatch(stagingAreaBatchIdAdded({ batchId }));
      } catch {
        if (didInitializeStagingArea) {
          // We initialized the staging area in this listener, and there was a problem at some point. This means
          // there only possible canvas batch id is the one we just added, so we can reset the staging area without
          // losing any data.
          dispatch(stagingAreaReset());
        }
      }
    },
  });
};
