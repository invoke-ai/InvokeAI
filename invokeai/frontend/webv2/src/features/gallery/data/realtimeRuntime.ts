import type { BackendConnectionStatus } from '@platform/transport/types';

import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';

export interface GalleryRealtimeBackend {
  on(event: string, handler: (payload: never) => void): () => void;
  onConnectionChange(handler: (status: BackendConnectionStatus) => void): () => void;
}

export interface GalleryRealtimeRuntime {
  dispose(): void;
  start(): void;
}

/**
 * Refresh external uploads and reconnects at least minIntervalMs apart. Coalesce pending events and guarantee a
 * trailing pass without repeatedly cancelling page fetches.
 */
export const createGalleryRealtimeRuntime = ({
  backend,
  coalesceMs = 250,
  invalidate,
  minIntervalMs = 1000,
}: {
  backend: GalleryRealtimeBackend;
  coalesceMs?: number;
  invalidate: () => void | Promise<void>;
  minIntervalMs?: number;
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
