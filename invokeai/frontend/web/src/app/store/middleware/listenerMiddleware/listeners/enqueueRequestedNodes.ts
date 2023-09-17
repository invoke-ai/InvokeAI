import { logger } from 'app/logging/logger';
import { enqueueRequested } from 'app/store/actions';
import { parseify } from 'common/util/serialize';
import { buildNodesGraph } from 'features/nodes/util/graphBuilders/buildNodesGraph';
import { addToast } from 'features/system/store/systemSlice';
import { t } from 'i18next';
import { queueApi } from 'services/api/endpoints/queue';
import { BatchConfig } from 'services/api/types';
import { startAppListening } from '..';

export const addEnqueueRequestedNodes = () => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof enqueueRequested> =>
      enqueueRequested.match(action) && action.payload.tabName === 'nodes',
    effect: async (action, { getState, dispatch }) => {
      const log = logger('queue');
      const state = getState();
      const { prepend } = action.payload;
      const graph = buildNodesGraph(state.nodes);
      const batchConfig: BatchConfig = {
        batch: {
          graph,
          runs: state.generation.iterations,
        },
        prepend: action.payload.prepend,
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
          queueApi.endpoints.resumeProcessor.initiate(undefined, {
            fixedCacheKey: 'resumeProcessor',
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
