import { createAction } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { parseify } from 'common/util/serialize';
import { buildAdHocUpscaleGraph } from 'features/nodes/util/graphBuilders/buildAdHocUpscaleGraph';
import { addToast } from 'features/system/store/systemSlice';
import { t } from 'i18next';
import { queueApi } from 'services/api/endpoints/queue';
import { startAppListening } from '..';

export const upscaleRequested = createAction<{ image_name: string }>(
  `upscale/upscaleRequested`
);

export const addUpscaleRequestedListener = () => {
  startAppListening({
    actionCreator: upscaleRequested,
    effect: async (action, { dispatch, getState }) => {
      const log = logger('session');

      const { image_name } = action.payload;
      const { esrganModelName } = getState().postprocessing;

      const graph = buildAdHocUpscaleGraph({
        image_name,
        esrganModelName,
      });

      try {
        const req = dispatch(
          queueApi.endpoints.enqueueGraph.initiate(
            { graph, prepend: true },
            {
              fixedCacheKey: 'enqueueGraph',
            }
          )
        );

        const enqueueResult = await req.unwrap();
        req.reset();
        dispatch(
          queueApi.endpoints.resumeProcessor.initiate(undefined, {
            fixedCacheKey: 'resumeProcessor',
          })
        );
        log.debug(
          { enqueueResult: parseify(enqueueResult) },
          t('queue.graphQueued')
        );
      } catch {
        log.error({ graph: parseify(graph) }, t('queue.graphFailedToQueue'));
        dispatch(
          addToast({
            title: t('queue.graphFailedToQueue'),
            status: 'error',
          })
        );
      }
    },
  });
};
