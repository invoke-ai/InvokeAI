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
      const {
        queue_item_id: item_id,
        queue_batch_id,
        status,
      } = action.payload.data;
      log.debug(
        action.payload,
        `Queue item ${item_id} status updated: ${status}`
      );
      dispatch(appSocketQueueItemStatusChanged(action.payload));

      dispatch(
        queueApi.util.updateQueryData('listQueueItems', undefined, (draft) => {
          queueItemsAdapter.updateOne(draft, {
            id: item_id,
            changes: action.payload.data,
          });
        })
      );

      dispatch(
        queueApi.util.invalidateTags([
          'CurrentSessionQueueItem',
          'NextSessionQueueItem',
          { type: 'SessionQueueItem', id: item_id },
          { type: 'SessionQueueItemDTO', id: item_id },
          { type: 'BatchStatus', id: queue_batch_id },
        ])
      );

      const req = dispatch(
        queueApi.endpoints.getQueueStatus.initiate(undefined, {
          forceRefetch: true,
        })
      );
      await req.unwrap();
      req.unsubscribe();
    },
  });
};
