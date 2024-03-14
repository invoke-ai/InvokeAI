import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { queueApi, queueItemsAdapter } from 'services/api/endpoints/queue';
import { socketQueueItemStatusChanged } from 'services/events/actions';

const log = logger('socketio');

export const addSocketQueueItemStatusChangedEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketQueueItemStatusChanged,
    effect: async (action, { dispatch }) => {
      // we've got new status for the queue item, batch and queue
      const { item_id, status, started_at, updated_at, error, completed_at, created_at, batch_status, queue_status } =
        action.payload.data;

      log.debug(action.payload, `Queue item ${item_id} status updated: ${status}`);

      // Update this specific queue item in the list of queue items (this is the queue item DTO, without the session)
      dispatch(
        queueApi.util.updateQueryData('listQueueItems', undefined, (draft) => {
          queueItemsAdapter.updateOne(draft, {
            id: String(item_id),
            changes: {
              status,
              started_at,
              updated_at: updated_at ?? undefined,
              error,
              completed_at: completed_at ?? undefined,
            },
          });
        })
      );

      // Update the queue status (we do not get the processor status here)
      dispatch(
        queueApi.util.updateQueryData('getQueueStatus', undefined, (draft) => {
          if (!draft) {
            return;
          }
          Object.assign(draft.queue, queue_status);
        })
      );

      // Update the batch status
      dispatch(
        queueApi.util.updateQueryData('getBatchStatus', { batch_id: batch_status.batch_id }, () => batch_status)
      );

      // Update the queue item status (this is the full queue item, including the session)
      dispatch(
        queueApi.util.updateQueryData('getQueueItem', item_id, (draft) => {
          if (!draft) {
            return;
          }
          Object.assign(draft, {
            status,
            started_at,
            updated_at,
            error,
            completed_at,
            created_at,
          });
        })
      );

      // Invalidate caches for things we cannot update
      // TODO: technically, we could possibly update the current session queue item, but feels safer to just request it again
      dispatch(
        queueApi.util.invalidateTags(['CurrentSessionQueueItem', 'NextSessionQueueItem', 'InvocationCacheStatus'])
      );
    },
  });
};
