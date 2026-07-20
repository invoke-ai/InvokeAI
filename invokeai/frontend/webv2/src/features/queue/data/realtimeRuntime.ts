import type { QueueBackendPort } from '@features/queue/core/types';
import type {
  InvocationProgressEvent,
  InvocationStartedEvent,
  QueueItemStatusChangedEvent,
} from '@features/queue/data/events';

import { isTerminalBackendStatus } from '@features/queue/data/events';

export interface QueueItemProgressPort {
  clear(itemId: number): void;
  clearAll(): void;
  set(
    itemId: number,
    progress: {
      image?: { dataUrl: string; height: number; width: number };
      message: string;
      percentage: number | null;
    }
  ): void;
}

export interface QueueRealtimeRuntime {
  dispose(): void;
  start(): void;
}

/**
 * Owns the global queue read model's one realtime subscription. Event bursts
 * collapse into one invalidation; progress frames update their transient store
 * directly and never force the full queue list through React state.
 */
export const createQueueRealtimeRuntime = ({
  backend,
  coalesceMs = 50,
  invalidate,
  progress,
  refreshModelCache,
}: {
  backend: Pick<QueueBackendPort, 'on' | 'onConnectionChange'>;
  coalesceMs?: number;
  invalidate: () => void | Promise<void>;
  progress: QueueItemProgressPort;
  refreshModelCache: () => void | Promise<void>;
}): QueueRealtimeRuntime => {
  const detachers: Array<() => void> = [];
  let invalidationTimer: ReturnType<typeof setTimeout> | null = null;
  let isStarted = false;

  const scheduleInvalidation = (): void => {
    if (invalidationTimer !== null) {
      return;
    }

    invalidationTimer = setTimeout(() => {
      invalidationTimer = null;
      void invalidate();
    }, coalesceMs);
  };

  const start = (): void => {
    if (isStarted) {
      return;
    }

    isStarted = true;
    void refreshModelCache();
    scheduleInvalidation();

    detachers.push(
      backend.on('queue_item_status_changed', (payload: never) => {
        const event = payload as unknown as QueueItemStatusChangedEvent;

        if (isTerminalBackendStatus(event.status) && event.status !== 'completed') {
          progress.clear(event.item_id);
        }

        scheduleInvalidation();
      }),
      backend.on('batch_enqueued', scheduleInvalidation),
      backend.on('queue_cleared', scheduleInvalidation),
      backend.on('queue_items_retried', scheduleInvalidation),
      backend.on('queue_items_canceled', scheduleInvalidation),
      backend.on('invocation_started', (payload: never) => {
        const event = payload as unknown as InvocationStartedEvent;

        progress.set(event.item_id, { message: '', percentage: null });
      }),
      backend.on('invocation_progress', (payload: never) => {
        const event = payload as unknown as InvocationProgressEvent;

        progress.set(event.item_id, {
          image: event.image?.dataURL
            ? { dataUrl: event.image.dataURL, height: event.image.height, width: event.image.width }
            : undefined,
          message: event.message,
          percentage: event.percentage,
        });
      }),
      backend.on('model_load_complete', () => {
        void refreshModelCache();
      }),
      backend.onConnectionChange((status) => {
        if (status === 'connected') {
          progress.clearAll();
          scheduleInvalidation();
          void refreshModelCache();
        }
      })
    );
  };

  const dispose = (): void => {
    for (const detach of detachers.splice(0)) {
      detach();
    }

    if (invalidationTimer !== null) {
      clearTimeout(invalidationTimer);
      invalidationTimer = null;
    }

    isStarted = false;
  };

  return { dispose, start };
};
