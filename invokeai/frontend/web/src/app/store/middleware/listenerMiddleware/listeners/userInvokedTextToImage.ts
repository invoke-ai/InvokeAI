import { logger } from 'app/logging/logger';
import { enqueueRequested } from 'app/store/actions';
import { parseify } from 'common/util/serialize';
import { buildLinearSDXLTextToImageGraph } from 'features/nodes/util/graphBuilders/buildLinearSDXLTextToImageGraph';
import { buildLinearTextToImageGraph } from 'features/nodes/util/graphBuilders/buildLinearTextToImageGraph';
import { addToast } from 'features/system/store/systemSlice';
import { t } from 'i18next';
import { queueApi } from 'services/api/endpoints/queue';
import { startAppListening } from '..';

export const addUserInvokedTextToImageListener = () => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof enqueueRequested> =>
      enqueueRequested.match(action) && action.payload.tabName === 'txt2img',
    effect: async (action, { getState, dispatch }) => {
      const log = logger('session');
      const state = getState();
      const model = state.generation.model;
      const { prepend } = action.payload;

      let graph;

      if (model && model.base_model === 'sdxl') {
        graph = buildLinearSDXLTextToImageGraph(state);
      } else {
        const batchConfig = buildLinearTextToImageGraph(state, prepend);

        try {
          const req = dispatch(
            queueApi.endpoints.enqueueBatch.initiate(batchConfig, {
              fixedCacheKey: 'enqueueBatch',
            })
          );

          const enqueueResult = await req.unwrap();
          req.reset();
          dispatch(queueApi.endpoints.startQueueExecution.initiate());

          log.debug(
            { enqueueResult: parseify(enqueueResult) },
            'Text to Image batch enqueued'
          );
          dispatch(
            addToast({
              title: t('queue.batchQueued'),
              description: t('queue.batchQueuedDesc', {
                count: enqueueResult.enqueued,
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
      }

      // dispatch(textToImageGraphBuilt(graph));

      // log.debug({ graph: parseify(graph) }, 'Text to Image graph built');

      // dispatch(sessionCreated({ graph }));

      // await take(sessionCreated.fulfilled.match);

      // dispatch(sessionReadyToInvoke());
    },
  });
};
