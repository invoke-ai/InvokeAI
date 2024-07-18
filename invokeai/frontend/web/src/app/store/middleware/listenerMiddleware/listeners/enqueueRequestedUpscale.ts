import { enqueueRequested } from 'app/store/actions';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { isImageViewerOpenChanged } from 'features/gallery/store/gallerySlice';
import { prepareLinearUIBatch } from 'features/nodes/util/graph/buildLinearBatchConfig';
import { queueApi } from 'services/api/endpoints/queue';
import { createIsAllowedToUpscaleSelector } from '../../../../../features/parameters/hooks/useIsAllowedToUpscale';
import { toast } from '../../../../../features/toast/toast';
import { t } from 'i18next';
import { logger } from '../../../../logging/logger';
import { useAppSelector } from '../../../storeHooks';
import { buildMultidiffusionUpscsaleGraph } from '../../../../../features/nodes/util/graph/buildMultidiffusionUpscaleGraph';

export const addEnqueueRequestedUpscale = (startAppListening: AppStartListening) => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof enqueueRequested> =>
      enqueueRequested.match(action) && action.payload.tabName === 'upscaling',
    effect: async (action, { getState, dispatch }) => {
      console.log("in listener")
      const log = logger('session');

      const state = getState();
      const { shouldShowProgressInViewer } = state.ui;
      const model = state.generation.model;
      const { prepend } = action.payload;


      const graph = await buildMultidiffusionUpscsaleGraph(state)



      // const batchConfig = prepareLinearUIBatch(state, graph, prepend);

      const req = dispatch(
        queueApi.endpoints.enqueueBatch.initiate({
          batch: {
            graph,
            runs: 1,
          },
        }, {
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
