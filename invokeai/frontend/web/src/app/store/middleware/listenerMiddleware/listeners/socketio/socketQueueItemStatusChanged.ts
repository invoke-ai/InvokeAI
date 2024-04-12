import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { queueApi, queueItemsAdapter } from 'services/api/endpoints/queue';
import { socketQueueItemStatusChanged } from 'services/events/actions';

const log = logger('socketio');

/**
 * Updates the page's title to reflect the current length of the active queue.
 * If there are items in the queue, the title will be prefixed with the count inside parentheses.
 * If the queue is empty, the title will revert to its original state without any count.
 *
 * @param {number} activeQueueLength - The number of active items in the queue.
 */
const updatePageTitle = (activeQueueLength: number) => {
  const baseString: string = document.title.replace(/\(\d+\)/ , '');
  document.title = activeQueueLength > 0 ? `(${activeQueueLength}) ${baseString}` : baseString;
};

/**
 * Updates the favicon of the page based on the presence of items in the queue.
 * If there are items in the queue, it uses an alert favicon. Otherwise, it reverts to the default favicon.
 *
 * @param {number} activeQueueLength - The number of active items in the queue.
 */
const updatePageFavicon = (activeQueueLength: number) => {
  const InvokeLogoSVG: string = 'assets/images/invoke-favicon.svg';
  const InvokeAlertLogoSVG: string = 'assets/images/invoke-alert-favicon.svg';
  const faviconEl: HTMLLinkElement = document.getElementById('invoke-favicon') as HTMLLinkElement;
  faviconEl.href = activeQueueLength > 0 ? InvokeAlertLogoSVG : InvokeLogoSVG;
};

export const addSocketQueueItemStatusChangedEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketQueueItemStatusChanged,
    effect: async (action, { dispatch }) => {
      // we've got new status for the queue item, batch and queue
      const { queue_item, batch_status, queue_status } = action.payload.data;

      // Keep track of the active queue length by summing up pending and in_progress count
      const activeQueueLength: number = (queue_status.pending + queue_status.in_progress) || 0;

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

      // Update the queue item status (this is the full queue item, including the session)
      dispatch(
        queueApi.util.updateQueryData('getQueueItem', queue_item.item_id, (draft) => {
          if (!draft) {
            return;
          }
          Object.assign(draft, queue_item);
        })
      );

      // Invalidate caches for things we cannot update
      // TODO: technically, we could possibly update the current session queue item, but feels safer to just request it again
      dispatch(
        queueApi.util.invalidateTags(['CurrentSessionQueueItem', 'NextSessionQueueItem', 'InvocationCacheStatus'])
      );

      updatePageTitle(activeQueueLength);
      updatePageFavicon(activeQueueLength);
    },
  });
};
