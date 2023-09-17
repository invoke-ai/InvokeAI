import { logger } from 'app/logging/logger';
import { batchEnqueued } from 'app/store/actions';
import { parseify } from 'common/util/serialize';
import { addToast } from 'features/system/store/systemSlice';
import { t } from 'i18next';
import { queueApi } from 'services/api/endpoints/queue';
import { startAppListening } from '..';

export const addBatchEnqueuedListener = () => {
  startAppListening({
    actionCreator: batchEnqueued,
    effect: async (action, { dispatch }) => {
      const log = logger('session');
      const batchConfig = action.payload;
      const { prepend } = batchConfig;

      try {
        const req = dispatch(
          queueApi.endpoints.enqueueBatch.initiate(batchConfig, {
            fixedCacheKey: 'enqueueBatch',
          })
        );
        const enqueueResult = await req.unwrap();
        req.reset();

        dispatch(
          queueApi.endpoints.resumeProcessor.initiate(undefined, {
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
