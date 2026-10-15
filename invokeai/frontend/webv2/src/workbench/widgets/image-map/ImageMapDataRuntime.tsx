import type { ImageIndexStatusEvent, ImageMapProjectionReadyEvent } from '@workbench/image-map/events';

import { getAuthSession } from '@features/identity';
import { useMountEffect } from '@platform/react/useMountEffect';
import { socketHub } from '@platform/transport/socketHub';
import {
  imageMapStore,
  recordImageIndexStatus,
  refreshImageIndexStatus,
  refreshImageMapPoints,
} from '@workbench/image-map/imageMapStore';

/**
 * Refresh loaded maps on projection events and reconnects; feed footer counts without fetching for sessions that
 * never open the map.
 */
/**
 * An event naming a user other than the session's. Admins receive index and
 * projection events for every user, and an admin refetch usually finds its own
 * all-images projection stale and enqueues another full recompute — so one
 * user's activity would drive UMAP fits on every admin's client. Unknown on
 * either side is not evidence of a foreign event: single-user mode has no
 * session user at all, and an event without a user id predates the field.
 */
const isForAnotherUser = (userId: unknown): boolean => {
  const sessionUserId = getAuthSession().user?.user_id;

  return typeof userId === 'string' && typeof sessionUserId === 'string' && userId !== sessionUserId;
};

export const attachImageMapDataRuntime = (): (() => void) => {
  // The widget host is mounted whenever the image-map widget is *available*,
  // not when it is open, so anything unconditional here bills a user who never
  // looks at the map.
  const hasLoadedOnce = () => {
    const { loadState } = imageMapStore.getSnapshot();

    return loadState === 'loaded' || loadState === 'error';
  };

  const refreshIfLoaded = () => {
    const { loadState, renderError } = imageMapStore.getSnapshot();

    // Ignore ordinary events during first load; projection-ready is the exception because it may supersede the
    // in-flight response.
    if (loadState !== 'loaded' && loadState !== 'error') {
      return;
    }

    // A failed canvas is not something fresh data can repair, and every
    // successful refresh clears `renderError` — which remounts the plot, fails
    // again, and flickers once per event for the length of a backfill.
    // Clearing it stays what it was designed to be: the user's deliberate retry.
    if (renderError) {
      return;
    }

    void refreshImageMapPoints();
  };

  // Allow projection-ready during loading: recompute can overtake the fetch and invalidate labels. Store dedupe
  // queues at most one rerun.
  const refreshProjectionReady = () => {
    const { loadState, renderError } = imageMapStore.getSnapshot();

    if (renderError || (loadState !== 'loaded' && loadState !== 'error' && loadState !== 'loading')) {
      return;
    }

    void refreshImageMapPoints();
  };

  const detachers = [
    // The backend routes this to the requesting user's room plus admins, so
    // receipt alone does not mean "my map changed".
    socketHub.on('image_map_projection_ready', (payload: never) => {
      if (!isForAnotherUser((payload as unknown as ImageMapProjectionReadyEvent | undefined)?.user_id)) {
        refreshProjectionReady();
      }
    }),
    // Counts-free per-user poke: the owner's images just reached the index.
    // This is the only index signal non-admins receive (status events below
    // are admin-only), so it is what makes a non-admin's map follow their
    // own generations.
    socketHub.on('image_index_updated', refreshIfLoaded),
    socketHub.on('image_index_status', (payload: never) => {
      const event = payload as unknown as ImageIndexStatusEvent;
      recordImageIndexStatus({
        embedded: event.embedded,
        failed: event.failed ?? 0,
        pending: event.pending,
        total: event.total,
      });
      // At indexing quiescence, refresh to trigger scope-hash comparison and needed recomputation. Admin counts
      // cover all users; non-admin embed pokes cover their own images, while deletions may need manual refresh.
      if (event.pending === 0) {
        refreshIfLoaded();
      }
    }),
  ];
  // Use connection replay only as baseline; refresh on actual disconnected-to-connected transitions.
  let previousStatus: string | null = null;
  const detachConnection = socketHub.onConnectionChange((status) => {
    const isReconnect = previousStatus !== null && previousStatus !== 'connected' && status === 'connected';

    previousStatus = status;

    if (isReconnect) {
      refreshIfLoaded();

      // Reread counts after reconnect because status events are not replayed; retain the map-open guard.
      if (hasLoadedOnce()) {
        refreshImageIndexStatus();
      }
    }
  });

  return () => {
    for (const detach of detachers) {
      detach();
    }
    detachConnection();
  };
};

/** React is only the idempotent lifecycle adapter for the non-React runtime. */
export const ImageMapDataRuntime = () => {
  useMountEffect(attachImageMapDataRuntime);

  return null;
};
