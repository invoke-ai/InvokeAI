import { logger } from 'app/logging/logger';
import { enqueueRequested } from 'app/store/actions';
import { parseify } from 'common/util/serialize';
import { buildLinearSDXLTextToImageGraph } from 'features/nodes/util/graphBuilders/buildLinearSDXLTextToImageGraph';
import { buildLinearTextToImageGraph } from 'features/nodes/util/graphBuilders/buildLinearTextToImageGraph';
import { addToast } from 'features/system/store/systemSlice';
import { t } from 'i18next';
import { queueApi } from 'services/api/endpoints/queue';
import { startAppListening } from '..';
import { prepareLinearUIBatch } from 'features/nodes/util/graphBuilders/buildLinearBatchConfig';
import { buildLinearSDXLImageToImageGraph } from 'features/nodes/util/graphBuilders/buildLinearSDXLImageToImageGraph';
import { buildLinearImageToImageGraph } from 'features/nodes/util/graphBuilders/buildLinearImageToImageGraph';

export const addUserEnqueuedT2iOrI2iListener = () => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof enqueueRequested> =>
      enqueueRequested.match(action) &&
      (action.payload.tabName === 'txt2img' ||
        action.payload.tabName === 'img2img'),
    effect: async (action, { getState, dispatch }) => {
      const log = logger('session');
      const state = getState();
      const model = state.generation.model;
      const { prepend } = action.payload;

      let graph;

      if (model && model.base_model === 'sdxl') {
        if (action.payload.tabName === 'txt2img') {
          graph = buildLinearSDXLTextToImageGraph(state);
        } else {
          graph = buildLinearSDXLImageToImageGraph(state);
        }
      } else {
        if (action.payload.tabName === 'txt2img') {
          graph = buildLinearTextToImageGraph(state);
        } else {
          graph = buildLinearImageToImageGraph(state);
        }
      }

      const batchConfig = prepareLinearUIBatch(state, graph, prepend);

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
        dispatch(
          addToast({
            title: t('queue.batchQueued'),
            description: t('queue.batchQueuedDesc', {
              item_count: enqueueResult.enqueued,
              direction: prepend ? t('queue.front') : t('queue.back'),
              directionAction: prepend
                ? t('queue.prepended')
                : t('queue.appended'),
            }),
            status: 'success',
          })
        );
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
