import { createAction } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { parseify } from 'common/util/serialize';
import { buildAdHocUpscaleGraph } from 'features/nodes/util/graphBuilders/buildAdHocUpscaleGraph';
import { addToast } from 'features/system/store/systemSlice';
import { t } from 'i18next';
import { queueApi } from 'services/api/endpoints/queue';
import { BatchConfig } from 'services/api/types';
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
      const batchConfig: BatchConfig = {
        batch: {
          graph,
          runs: 1,
        },
        prepend: true,
      };

      try {
        const req = dispatch(
          queueApi.endpoints.enqueueBatch.initiate(batchConfig, {
            fixedCacheKey: 'enqueueBatch',
          })
        );

        const enqueueResult = await req.unwrap();
        req.reset();
        dispatch(
          queueApi.endpoints.startQueueExecution.initiate(undefined, {
            fixedCacheKey: 'startQueue',
          })
        );
        log.debug({ enqueueResult: parseify(enqueueResult) }, 'Batch enqueued');
      } catch {
        log.error(
          { batchConfig: parseify(batchConfig) },
          'Failed to enqueue batch'
        );
        dispatch(
          addToast({
            title: t('queue.batchFailedToQueue'),
            status: 'error',
          })
        );
      }
    },
  });
};
