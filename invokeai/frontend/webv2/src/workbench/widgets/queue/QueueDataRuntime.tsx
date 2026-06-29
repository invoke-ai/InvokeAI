import type {
  InvocationProgressEvent,
  InvocationStartedEvent,
  QueueItemStatusChangedEvent,
} from '@workbench/backend/events';

import { isTerminalBackendStatus } from '@workbench/backend/events';
import { itemProgressStore } from '@workbench/backend/itemProgressStore';
import { socketHub } from '@workbench/backend/socketHub';
import { ConfirmDialog } from '@workbench/components/ui';
import { useEffect } from 'react';

import { refreshModelCacheStats } from './modelCacheStore';
import { clearQueueConfirmation, useQueueConfirmation } from './queueConfirmationStore';
import { ensureQueueLoaded, refreshQueue, setQueueScope } from './queueDataStore';
import { useQueueQueryScope } from './queueScope';

/**
 * Singleton runtime for the Queue widget (mounted via `manifest.host`). It owns
 * the live wiring: socket lifecycle events refresh the server-queue store and
 * feed per-item progress, replacing the RTK Query tag-invalidation the legacy v6
 * app used. Pure side effects — renders nothing.
 */
export const attachQueueDataRuntime = (): (() => void) => {
  ensureQueueLoaded();
  void refreshModelCacheStats();

  const refresh = () => {
    void refreshQueue();
  };
  const detachers = [
    socketHub.on('queue_item_status_changed', (payload: never) => {
      const event = payload as unknown as QueueItemStatusChangedEvent;

      if (isTerminalBackendStatus(event.status) && event.status !== 'completed') {
        itemProgressStore.clear(event.item_id);
      }

      refresh();
    }),
    socketHub.on('batch_enqueued', refresh),
    socketHub.on('queue_cleared', refresh),
    socketHub.on('queue_items_retried', refresh),
    socketHub.on('invocation_started', (payload: never) => {
      const event = payload as unknown as InvocationStartedEvent;
      itemProgressStore.set(event.item_id, { message: '', percentage: null });
    }),
    socketHub.on('invocation_progress', (payload: never) => {
      const event = payload as unknown as InvocationProgressEvent;
      itemProgressStore.set(event.item_id, {
        image: event.image?.dataURL
          ? { dataUrl: event.image.dataURL, height: event.image.height, width: event.image.width }
          : undefined,
        message: event.message,
        percentage: event.percentage,
      });
    }),
    socketHub.on('model_load_complete', () => {
      void refreshModelCacheStats();
    }),
  ];

  const detachConnection = socketHub.onConnectionChange((status) => {
    if (status === 'connected') {
      itemProgressStore.clearAll();
      void refreshQueue();
      void refreshModelCacheStats();
    }
  });

  return () => {
    for (const detach of detachers) {
      detach();
    }

    detachConnection();
  };
};

export const QueueDataRuntime = () => {
  const scope = useQueueQueryScope();
  const confirmation = useQueueConfirmation();

  useEffect(() => {
    setQueueScope(scope);
  }, [scope, scope.originPrefix]);

  useEffect(() => {
    return attachQueueDataRuntime();
  }, []);

  return confirmation ? (
    <ConfirmDialog
      body={confirmation.body}
      confirmLabel={confirmation.confirmLabel}
      isOpen={true}
      title={confirmation.title}
      onClose={clearQueueConfirmation}
      onConfirm={confirmation.onConfirm}
    />
  ) : null;
};
