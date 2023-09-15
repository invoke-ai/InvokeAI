import { logger } from 'app/logging/logger';
import { canvasSessionIdAdded } from 'features/canvas/store/canvasSlice';
import { queueApi, queueItemsAdapter } from 'services/api/endpoints/queue';
import {
  appSocketQueueItemStatusChanged,
  socketQueueItemStatusChanged,
} from 'services/events/actions';
import { startAppListening } from '../..';

export const addSocketQueueItemStatusChangedEventListener = () => {
  startAppListening({
    actionCreator: socketQueueItemStatusChanged,
    effect: (action, { dispatch, getState }) => {
      const log = logger('socketio');
      const { item_id, status: newStatus } = action.payload.data;
      log.debug(
        action.payload,
        `Queue item ${item_id} status updated: ${newStatus}`
      );
      dispatch(appSocketQueueItemStatusChanged(action.payload));

      dispatch(
        queueApi.util.updateQueryData('listQueueItems', undefined, (draft) => {
          queueItemsAdapter.updateOne(draft, {
            id: item_id,
            changes: { status: newStatus },
          });
        })
      );

      const state = getState();
      const { batch_id, graph_execution_state_id } = action.payload.data;
      if (state.canvas.batchIds.includes(batch_id)) {
        dispatch(canvasSessionIdAdded(graph_execution_state_id));
      }

      dispatch(
        queueApi.util.invalidateTags([
          'CurrentSessionQueueItem',
          'NextSessionQueueItem',
          'SessionQueueStatus',
        ])
      );
    },
  });
};
