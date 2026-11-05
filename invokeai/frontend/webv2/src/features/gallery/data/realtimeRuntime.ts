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
 * Refresh uploads, all of the session user's completions, and reconnects at least minIntervalMs apart. Hidden
 * tabs retain one pending pass until visible. The Query invalidation owner serializes slow refetches.
 */
export const createGalleryRealtimeRuntime = ({
  backend,
  coalesceMs = 250,
  getSessionUserId,
  invalidate,
  isClientQueueItemOrigin,
  isVisible = () => true,
  minIntervalMs = 1000,
  onRevealRequest,
  onVisibilityChange,
}: {
  backend: GalleryRealtimeBackend;
  coalesceMs?: number;
  /** Null when events cannot be attributed to a user (single-user servers tag every event `system`). */
  getSessionUserId: () => string | null;
  invalidate: () => void | Promise<void>;
  /** Frontend origins refresh only; they never request external reveal. */
  isClientQueueItemOrigin: (origin: string | null) => boolean;
  isVisible?: () => boolean;
  minIntervalMs?: number;
  onVisibilityChange?: (listener: () => void) => () => void;
  /** Receives the backend item id of each of those completions whose batch asked to be revealed. */
  onRevealRequest?: (queueItemId: number) => void;
}): GalleryRealtimeRuntime => {
  const owner = captureAccountScope();
  const detachers: Array<() => void> = [];
  let invalidationTimer: ReturnType<typeof setTimeout> | null = null;
  let lastInvalidatedAt = Number.NEGATIVE_INFINITY;
  let isStarted = false;
  let invalidationPending = false;
  const isActive = (): boolean => isStarted && isAccountScopeCurrent(owner);

  const scheduleInvalidation = (): void => {
    if (!isActive() || !invalidationPending || !isVisible() || invalidationTimer !== null) {
      return;
    }

    // Use monotonic time so wall-clock corrections cannot extend the pending timer and delay all subsequent
    // events.
    const delay = Math.max(coalesceMs, lastInvalidatedAt + minIntervalMs - performance.now());

    invalidationTimer = setTimeout(() => {
      invalidationTimer = null;

      if (isActive() && isVisible()) {
        invalidationPending = false;
        lastInvalidatedAt = performance.now();
        void invalidate();
      }
    }, delay);
  };

  const requestInvalidation = (): void => {
    if (isActive()) {
      invalidationPending = true;
      scheduleInvalidation();
    }
  };

  const handleVisibilityChange = (): void => {
    if (!isVisible() && invalidationTimer !== null) {
      clearTimeout(invalidationTimer);
      invalidationTimer = null;
    }
    scheduleInvalidation();
  };

  const handleQueueItemStatusChanged = (event: QueueItemStatusChangedPayload): void => {
    // Admins receive other users' events too. All own-account completions refresh here, including other tabs.
    const sessionUserId = getSessionUserId();

    if (!isActive() || event.status !== 'completed' || (sessionUserId !== null && event.user_id !== sessionUserId)) {
      return;
    }

    requestInvalidation();

    if (!isClientQueueItemOrigin(event.origin) && event.destination === REVEAL_DESTINATION) {
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
      backend.on('image_uploaded', requestInvalidation),
      backend.on('video_uploaded', requestInvalidation),
      backend.on('queue_item_status_changed', handleQueueItemStatusChanged),
      backend.onConnectionChange((status) => {
        const isReconnect = previousStatus === 'disconnected' && status === 'connected';

        previousStatus = status;

        if (isReconnect) {
          requestInvalidation();
        }
      })
    );
    if (onVisibilityChange) {
      detachers.push(onVisibilityChange(handleVisibilityChange));
    }
  };

  const dispose = (): void => {
    isStarted = false;
    invalidationPending = false;

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
