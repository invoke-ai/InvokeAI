import { logger } from 'app/logging/logger';
import type { AppDispatch } from 'app/store/store';
import ErrorToastDescription, { getTitle } from 'features/toast/ErrorToastDescription';
import { toast } from 'features/toast/toast';
import type { ApiTagDescription } from 'services/api';
import { queueApi } from 'services/api/endpoints/queue';
import type { S } from 'services/api/types';
import { REDACTED_USER_ID } from 'services/events/eventScope';
import { QUEUE_CHANGED_TAGS } from 'services/events/queueCacheTags';
import { getUpdatedQueueStatusOnQueueItemStatusChanged } from 'services/events/queueStatusEvents';
import { $lastProgressEvent } from 'services/events/stores';

const log = logger('events');

type QueueItemStatusChangedEvent = S['QueueItemStatusChangedEvent'];

type Coordinator = {
  onQueueItemStatusChanged: (data: QueueItemStatusChangedEvent) => boolean;
};

/**
 * Builds the handler for the current user's own queue_item_status_changed events: optimistic
 * queue cache updates, tag invalidation, the failure toast, and clearing the progress indicator
 * on terminal statuses. The listener routes non-owner events to
 * `buildOnNonOwnerQueueItemStatusChanged` instead, so this handler never sees them.
 */
export const buildOnQueueItemStatusChanged = (dispatch: AppDispatch, coordinator: Coordinator) => {
  return (data: QueueItemStatusChangedEvent) => {
    if (!coordinator.onQueueItemStatusChanged(data)) {
      return;
    }

    // we've got new status for the queue item, batch and queue
    const {
      item_id,
      status,
      status_sequence,
      batch_status,
      error_type,
      error_message,
      destination,
      started_at,
      updated_at,
      completed_at,
      error_traceback,
    } = data;

    log.debug({ data }, `Queue item ${item_id} status updated: ${status}`);

    // Update this specific queue item in the list of queue items
    dispatch(
      queueApi.util.updateQueryData('getQueueItem', item_id, (draft) => {
        draft.status = status;
        draft.status_sequence = status_sequence;
        draft.started_at = started_at;
        draft.updated_at = updated_at;
        draft.completed_at = completed_at;
        draft.error_type = error_type;
        draft.error_message = error_message;
        draft.error_traceback = error_traceback;
      })
    );

    // Optimistically update the listAllQueueItems cache for this destination so the canvas
    // staging area immediately reflects status changes without waiting for a tag-based refetch
    if (destination) {
      dispatch(
        queueApi.util.updateQueryData('listAllQueueItems', { destination }, (draft) => {
          const item = draft.find((i) => i.item_id === item_id);
          if (item) {
            item.status = status;
            item.status_sequence = status_sequence;
            item.started_at = started_at;
            item.updated_at = updated_at;
            item.completed_at = completed_at;
            item.error_type = error_type;
            item.error_message = error_message;
            item.error_traceback = error_traceback;
          }
        })
      );
    }
    dispatch(
      queueApi.util.updateQueryData('getQueueStatus', undefined, (draft) =>
        getUpdatedQueueStatusOnQueueItemStatusChanged(draft, data)
      )
    );

    // Invalidate caches for things we cannot easily update
    // Invalidate SessionQueueStatus to refetch with user-specific counts
    const tagsToInvalidate: ApiTagDescription[] = [
      ...QUEUE_CHANGED_TAGS,
      'InvocationCacheStatus',
      { type: 'SessionQueueItem', id: item_id },
      { type: 'BatchStatus', id: batch_status.batch_id },
    ];
    if (destination) {
      tagsToInvalidate.push({ type: 'QueueCountsByDestination', id: destination });
    }
    dispatch(queueApi.util.invalidateTags(tagsToInvalidate));

    if (status === 'completed' || status === 'failed' || status === 'canceled') {
      if (status === 'failed' && error_type) {
        toast({
          id: `INVOCATION_ERROR_${error_type}`,
          title: getTitle(error_type),
          status: 'error',
          duration: null,
          updateDescription: true,
          description: <ErrorToastDescription errorType={error_type} errorMessage={error_message} />,
        });
      }
      // If the queue item is completed, failed, or cancelled, we want to clear the last progress event
      $lastProgressEvent.set(null);
    }
  };
};

/**
 * Builds the handler for queue_item_status_changed events that are not the current user's own.
 * Two kinds land here in multiuser mode:
 *
 * 1. The sanitized companion the backend broadcasts to other queue subscribers, with
 *    user_id="redacted" and identifiers/error fields cleared.
 * 2. Another user's full event, received by admins via the "admin" room (the backend skips owner
 *    and admin sids when broadcasting the sanitized companion, so admins only ever get the full
 *    one).
 *
 * In neither case may we run payload-driven cache mutations or per-session side effects — node
 * state reset, progress clear, completion bookkeeping, workflow reconciliation, or the failure
 * toast — those belong to the owner. Cache invalidation is the only permitted effect, but it must
 * cover everything the owner path covers: admin UI also renders the current/next queue item (the
 * cancel-current button), batch statuses, and the destination queue counts that back the canvas,
 * and for admins those queries span all users' items. The admin-room full event carries real ids,
 * so its batch and destination caches are invalidated narrowly; the sanitized companion redacts
 * batch_id and nulls destination, so only it falls back to the type-level tags.
 */
export const buildOnNonOwnerQueueItemStatusChanged = (dispatch: AppDispatch) => {
  return (data: QueueItemStatusChangedEvent) => {
    log.trace({ data }, `Non-owner queue_item_status_changed for item ${data.item_id}`);
    const tags: ApiTagDescription[] = [
      ...QUEUE_CHANGED_TAGS,
      'InvocationCacheStatus',
      { type: 'SessionQueueItem', id: data.item_id },
    ];
    if (data.user_id === REDACTED_USER_ID) {
      tags.push('BatchStatus', 'QueueCountsByDestination');
    } else {
      tags.push({ type: 'BatchStatus', id: data.batch_status.batch_id });
      if (data.destination) {
        tags.push({ type: 'QueueCountsByDestination', id: data.destination });
      }
    }
    dispatch(queueApi.util.invalidateTags(tags));
  };
};
