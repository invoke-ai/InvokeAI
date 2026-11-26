import type { QueueBackendPort } from '@features/queue/core/types';
import type {
  InvocationProgressEvent,
  InvocationStartedEvent,
  QueueItemStatusChangedEvent,
} from '@features/queue/data/events';
import type { QueueItemStatusChange } from '@features/queue/data/queries';

import { isTerminalBackendStatus } from '@features/queue/data/events';
import { mapQueueItemStatusChangedEvent } from '@features/queue/data/mappers';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';

export interface QueueItemProgressPort {
  clear(itemId: number): void;
  clearAll(): void;
  set(
    itemId: number,
    progress: {
      /** The accelerator running this session, e.g. `cuda:1` or `xpu:1`; null on unindexed and single-device installs. */
      device?: string | null;
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
 * Own one realtime subscription. Status changes apply to the read model at once and invalidate it, coalesced, to
 * reconcile; progress frames update transient stores directly.
 */
export const createQueueRealtimeRuntime = ({
  applyStatusChange,
  backend,
  coalesceMs = 50,
  invalidate,
  progress,
  refreshModelCache,
}: {
  /** Applies one item's status change to the loaded read models, which fence it against what they already hold. */
  applyStatusChange: (change: QueueItemStatusChange) => void;
  backend: Pick<QueueBackendPort, 'on' | 'onConnectionChange'>;
  coalesceMs?: number;
  invalidate: () => void | Promise<void>;
  progress: QueueItemProgressPort;
  refreshModelCache: () => void | Promise<void>;
}): QueueRealtimeRuntime => {
  const owner = captureAccountScope();
  const detachers: Array<() => void> = [];
  const latestStatusSequences = new Map<number, number>();
  let invalidationTimer: ReturnType<typeof setTimeout> | null = null;
  let isStarted = false;
  const isActive = (): boolean => isStarted && isAccountScopeCurrent(owner);

  const scheduleInvalidation = (): void => {
    if (!isActive() || invalidationTimer !== null) {
      return;
    }

    invalidationTimer = setTimeout(() => {
      invalidationTimer = null;

      if (isActive()) {
        void invalidate();
      }
    }, coalesceMs);
  };

  const start = (): void => {
    if (isStarted || !isAccountScopeCurrent(owner)) {
      return;
    }

    isStarted = true;
    void refreshModelCache();
    scheduleInvalidation();

    detachers.push(
      backend.on('queue_item_status_changed', (payload: never) => {
        if (!isActive()) {
          return;
        }

        const event = payload as unknown as QueueItemStatusChangedEvent;
        const sequence = event.status_sequence ?? null;
        const previousSequence = latestStatusSequences.get(event.item_id);
        const isStaleStatusSequence =
          sequence !== null && previousSequence !== undefined && sequence < previousSequence;

        if (!isStaleStatusSequence && sequence !== null) {
          if (isTerminalBackendStatus(event.status)) {
            latestStatusSequences.delete(event.item_id);
          } else {
            latestStatusSequences.set(event.item_id, sequence);
          }
        }

        // Requeued items must not keep a prior attempt's percentage; skip older status sequences so late pending events cannot blank active progress.
        if (
          !isStaleStatusSequence &&
          ((isTerminalBackendStatus(event.status) && event.status !== 'completed') ||
            event.status === 'pending' ||
            event.status === 'waiting')
        ) {
          progress.clear(event.item_id);
        }

        if (event.queue_status) {
          applyStatusChange(mapQueueItemStatusChangedEvent(event));
        }

        scheduleInvalidation();
      }),
      backend.on('batch_enqueued', scheduleInvalidation),
      backend.on('queue_cleared', scheduleInvalidation),
      backend.on('queue_items_retried', scheduleInvalidation),
      backend.on('queue_items_canceled', scheduleInvalidation),
      backend.on('invocation_started', (payload: never) => {
        if (!isActive()) {
          return;
        }

        const event = payload as unknown as InvocationStartedEvent;

        progress.set(event.item_id, { message: '', percentage: null });
      }),
      backend.on('invocation_progress', (payload: never) => {
        if (!isActive()) {
          return;
        }

        const event = payload as unknown as InvocationProgressEvent;

        progress.set(event.item_id, {
          device: event.device,
          image: event.image?.dataURL
            ? { dataUrl: event.image.dataURL, height: event.image.height, width: event.image.width }
            : undefined,
          message: event.message,
          percentage: event.percentage,
        });
      }),
      backend.on('model_load_complete', () => {
        if (isActive()) {
          void refreshModelCache();
        }
      }),
      backend.onConnectionChange((status) => {
        if (!isActive()) {
          return;
        }

        if (status === 'connected') {
          latestStatusSequences.clear();
          progress.clearAll();
          scheduleInvalidation();
          void refreshModelCache();
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

    latestStatusSequences.clear();
    progress.clearAll();
  };

  return { dispose, start };
};
