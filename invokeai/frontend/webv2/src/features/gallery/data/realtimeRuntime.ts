import type { BackendConnectionStatus } from '@platform/transport/types';

import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';

export interface GalleryRealtimeBackend {
  on(event: string, handler: (payload: never) => void): () => void;
  onConnectionChange(handler: (status: BackendConnectionStatus) => void): () => void;
}

/** The fields of the backend's `queue_item_status_changed` payload that Gallery reads. */
interface QueueItemStatusChangedPayload {
  destination: string | null;
  item_id: number;
  origin: string | null;
  status: string;
  user_id: string;
}

/**
 * External-client reveal contract. A batch enqueued with `POST /api/v1/queue/default/enqueue_batch` and
 * `batch.destination` set to this value, by the signed-in user (any user on a single-user server) and with an origin
 * that does not start with `webv2:`, asks for the last saved image of each completed item to be revealed, best
 * effort, in that user's open Gallery. Every other destination, `gallery` included (legacy web sends it on every
 * workflow and upscale run), only refreshes the Gallery. When the reveal happens, what it changes, and its limits
 * (workflow-call children among them) are documented in the Gallery README.
 */
const REVEAL_DESTINATION = 'gallery:reveal';

export interface GalleryRealtimeRuntime {
  dispose(): void;
  start(): void;
}

/**
 * Refresh external uploads, the session user's completions enqueued by other clients, and reconnects at least
 * minIntervalMs apart. Coalesce pending events and guarantee a trailing pass without repeatedly cancelling page
 * fetches.
 */
export const createGalleryRealtimeRuntime = ({
  backend,
  coalesceMs = 250,
  getSessionUserId,
  invalidate,
  isClientQueueItemOrigin,
  minIntervalMs = 1000,
  onRevealRequest,
}: {
  backend: GalleryRealtimeBackend;
  coalesceMs?: number;
  /** Null when events cannot be attributed to a user (single-user servers tag every event `system`). */
  getSessionUserId: () => string | null;
  invalidate: () => void | Promise<void>;
  /** True for origins of items this client enqueues; Queue refreshes for those in the tab that tracks them. */
  isClientQueueItemOrigin: (origin: string | null) => boolean;
  minIntervalMs?: number;
  /** Receives the backend item id of each of those completions whose batch asked to be revealed. */
  onRevealRequest?: (queueItemId: number) => void;
}): GalleryRealtimeRuntime => {
  const owner = captureAccountScope();
  const detachers: Array<() => void> = [];
  let invalidationTimer: ReturnType<typeof setTimeout> | null = null;
  let lastInvalidatedAt = Number.NEGATIVE_INFINITY;
  let isStarted = false;
  const isActive = (): boolean => isStarted && isAccountScopeCurrent(owner);

  const scheduleInvalidation = (): void => {
    if (!isActive() || invalidationTimer !== null) {
      return;
    }

    // Use monotonic time so wall-clock corrections cannot extend the pending timer and delay all subsequent
    // events.
    const delay = Math.max(coalesceMs, lastInvalidatedAt + minIntervalMs - performance.now());

    invalidationTimer = setTimeout(() => {
      invalidationTimer = null;

      if (isActive()) {
        lastInvalidatedAt = performance.now();
        void invalidate();
      }
    }, delay);
  };

  const handleQueueItemStatusChanged = (event: QueueItemStatusChangedPayload): void => {
    // Admins also receive other users' full events and everyone else's arrive redacted, so only the session user's
    // completions can add to this gallery. Items this client enqueued are left to Queue, which refreshes only in the
    // tab tracking the item: another tab or device of the same user is not refreshed for them. Known limitation,
    // left intentionally: refreshing here too would double-refresh the enqueuing tab.
    const sessionUserId = getSessionUserId();

    if (
      !isActive() ||
      event.status !== 'completed' ||
      isClientQueueItemOrigin(event.origin) ||
      (sessionUserId !== null && event.user_id !== sessionUserId)
    ) {
      return;
    }

    scheduleInvalidation();

    if (event.destination === REVEAL_DESTINATION) {
      onRevealRequest?.(event.item_id);
    }
  };

  const start = (): void => {
    if (isStarted || !isAccountScopeCurrent(owner)) {
      return;
    }

    isStarted = true;

    // Initial connection misses no events; only reconnects after disconnection should invalidate in-flight pages.
    // Subscription immediately replays status.
    let previousStatus: BackendConnectionStatus | null = null;

    detachers.push(
      backend.on('image_uploaded', scheduleInvalidation),
      backend.on('video_uploaded', scheduleInvalidation),
      backend.on('queue_item_status_changed', handleQueueItemStatusChanged),
      backend.onConnectionChange((status) => {
        const isReconnect = previousStatus === 'disconnected' && status === 'connected';

        previousStatus = status;

        if (isReconnect) {
          scheduleInvalidation();
        }
      })
    );
  };

  const dispose = (): void => {
    isStarted = false;

    for (const detach of detachers.splice(0)) {
      detach();
    }

    if (invalidationTimer !== null) {
      clearTimeout(invalidationTimer);
      invalidationTimer = null;
    }
  };

  return { dispose, start };
};
