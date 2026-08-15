import type { ImageIndexStatusEvent, ImageMapProjectionReadyEvent } from '@workbench/image-map/events';

import { getAuthSession } from '@features/identity';
import { useMountEffect } from '@platform/react/useMountEffect';
import { socketHub } from '@platform/transport/socketHub';
import { imageMapStore, refreshImageMapPoints } from '@workbench/image-map/imageMapStore';

/**
 * Socket-driven refresh for the image map, replacing polling: when the
 * backend announces a recomputed projection (or the socket reconnects after
 * an outage), the point set refetches. Index progress counts feed the footer.
 * Nothing fetches until the widget itself has loaded the map once — a user
 * who never opens the Image Map pays nothing.
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
  const refreshIfLoaded = () => {
    const { loadState, renderError } = imageMapStore.getSnapshot();

    // Only once the widget has actually loaded the map. `loading` used to pass
    // this guard, so an event landing during the first fetch queued a rerun
    // and forced a second full point set the moment it settled.
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

  const detachers = [
    // The backend routes this to the requesting user's room plus admins, so
    // receipt alone does not mean "my map changed".
    socketHub.on('image_map_projection_ready', (payload: never) => {
      if (!isForAnotherUser((payload as unknown as ImageMapProjectionReadyEvent | undefined)?.user_id)) {
        refreshIfLoaded();
      }
    }),
    // Counts-free per-user poke: the owner's images just reached the index.
    // This is the only index signal non-admins receive (status events below
    // are admin-only), so it is what makes a non-admin's map follow their
    // own generations.
    socketHub.on('image_index_updated', refreshIfLoaded),
    socketHub.on('image_index_status', (payload: never) => {
      const event = payload as unknown as ImageIndexStatusEvent;
      imageMapStore.patchSnapshot({
        indexCounts: {
          embedded: event.embedded,
          failed: event.failed ?? 0,
          pending: event.pending,
          total: event.total,
        },
      });
      // Quiescence is the moment the point set may have changed: batch
      // completions, deletions, and the final sweep after failures all end
      // in a pending === 0 emit (permanently-failed images are excluded
      // from pending so it always drains). Refetching makes the backend
      // compare scope hashes; if the map is stale it enqueues the recompute
      // whose projection-ready event closes the loop above. Without this
      // poke neither side ever initiates: the backend only recomputes when
      // asked, and nothing else asks. Status events are admin-only (the
      // counts aggregate every user's images); non-admins get the
      // image_index_updated poke above for their own embeds, and manual
      // refresh covers their deletions (the backend cannot resolve an
      // owner for an already-deleted row).
      if (event.pending === 0) {
        refreshIfLoaded();
      }
    }),
  ];
  const detachConnection = socketHub.onConnectionChange((status) => {
    if (status === 'connected') {
      refreshIfLoaded();
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
