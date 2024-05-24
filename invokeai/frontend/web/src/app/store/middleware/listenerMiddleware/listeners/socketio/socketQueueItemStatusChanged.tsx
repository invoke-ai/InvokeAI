import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { deepClone } from 'common/util/deepClone';
import { $nodeExecutionStates } from 'features/nodes/hooks/useExecutionState';
import { zNodeStatus } from 'features/nodes/types/invocation';
import ErrorToastDescription, { getTitleFromErrorType } from 'features/toast/ErrorToastDescription';
import { toast } from 'features/toast/toast';
import { forEach } from 'lodash-es';
import { queueApi, queueItemsAdapter } from 'services/api/endpoints/queue';
import { socketQueueItemStatusChanged } from 'services/events/actions';

const log = logger('socketio');

export const addSocketQueueItemStatusChangedEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketQueueItemStatusChanged,
    effect: async (action, { dispatch, getState }) => {
      // we've got new status for the queue item, batch and queue
      const { queue_item, batch_status, queue_status } = action.payload.data;

      log.debug(action.payload, `Queue item ${queue_item.item_id} status updated: ${queue_item.status}`);

      // Update this specific queue item in the list of queue items (this is the queue item DTO, without the session)
      dispatch(
        queueApi.util.updateQueryData('listQueueItems', undefined, (draft) => {
          queueItemsAdapter.updateOne(draft, {
            id: String(queue_item.item_id),
            changes: queue_item,
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

      // Invalidate caches for things we cannot update
      // TODO: technically, we could possibly update the current session queue item, but feels safer to just request it again
      dispatch(
        queueApi.util.invalidateTags([
          'CurrentSessionQueueItem',
          'NextSessionQueueItem',
          'InvocationCacheStatus',
          { type: 'SessionQueueItem', id: queue_item.item_id },
        ])
      );

      if (queue_item.status === 'in_progress') {
        forEach($nodeExecutionStates.get(), (nes) => {
          if (!nes) {
            return;
          }
          const clone = deepClone(nes);
          clone.status = zNodeStatus.enum.PENDING;
          clone.error = null;
          clone.progress = null;
          clone.progressImage = null;
          clone.outputs = [];
          $nodeExecutionStates.setKey(clone.nodeId, clone);
        });
      } else if (queue_item.status === 'failed' && queue_item.error_type) {
        const { error_type, error_message, session_id } = queue_item;
        const isLocal = getState().config.isLocal ?? true;
        const sessionId = session_id;

        toast({
          id: `INVOCATION_ERROR_${error_type}`,
          title: getTitleFromErrorType(error_type),
          status: 'error',
          duration: null,
          description: (
            <ErrorToastDescription
              errorType={error_type}
              errorMessage={error_message}
              sessionId={sessionId}
              isLocal={false}
            />
          ),
          updateDescription: isLocal ? true : false,
        });
      }
    },
  });
};
