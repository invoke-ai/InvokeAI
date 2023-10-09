import { logger } from 'app/logging/logger';
import { queueApi, queueItemsAdapter } from 'services/api/endpoints/queue';
import {
  appSocketQueueItemStatusChanged,
  socketQueueItemStatusChanged,
} from 'services/events/actions';
import { startAppListening } from '../..';

export const addSocketQueueItemStatusChangedEventListener = () => {
  startAppListening({
    actionCreator: socketQueueItemStatusChanged,
    effect: async (action, { dispatch }) => {
      const log = logger('socketio');

      const { queue_item, batch_status, queue_status } = action.payload.data;

      log.debug(
        action.payload,
        `Queue item ${queue_item.item_id} status updated: ${queue_item.status}`
      );
      dispatch(appSocketQueueItemStatusChanged(action.payload));

      dispatch(
        queueApi.util.updateQueryData('listQueueItems', undefined, (draft) => {
          queueItemsAdapter.updateOne(draft, {
            id: queue_item.item_id,
            changes: queue_item,
          });
        })
      );

      dispatch(
        queueApi.util.updateQueryData('getQueueStatus', undefined, (draft) => {
          Object.assign(draft.queue, queue_status);
        })
      );
      dispatch(
        queueApi.util.updateQueryData('getQueueStatus', undefined, (draft) => {
          if (!draft) {
            return;
          }
          Object.assign(draft.queue, queue_status);
        })
      );

      dispatch(
        queueApi.util.updateQueryData(
          'getBatchStatus',
          { batch_id: batch_status.batch_id },
          () => batch_status
        )
      );

      dispatch(
        queueApi.util.updateQueryData(
          'getQueueItem',
          queue_item.item_id,
          (draft) => {
            if (!draft) {
              return;
            }
            Object.assign(draft, queue_item);
          }
        )
      );

      dispatch(
        queueApi.util.invalidateTags([
          'CurrentSessionQueueItem',
          'NextSessionQueueItem',
          'InvocationCacheStatus',
        ])
      );
    },
  });
};
