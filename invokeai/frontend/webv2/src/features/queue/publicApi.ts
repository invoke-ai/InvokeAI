import type { QueueFeatureCommands, QueueQueryScope, QueueReadModel } from './core/types';
import type { QueueItemProgressPort, QueueRealtimeRuntime } from './data/realtimeRuntime';
import type {
  QueueHistoryPort,
  QueueResultDestinationPort,
  QueueRunJournalPort,
  QueueRunLockPort,
  QueueRuntime,
} from './runtime';
import type { QueueModelLoadPort, QueueNodeExecutionPort } from './runtime/coordinator';
import type { RemoteModelTransferToastPort } from './runtime/remoteModelTransferToasts';

import { queueBackend } from './data/httpRealtimeQueueBackend';
import { queueReadModelOptions } from './data/queries';
import { createQueueRealtimeRuntime } from './data/realtimeRuntime';
import { createQueueRuntime } from './runtime';
import { createQueueReceiptAcknowledgements, type QueueReceiptStorePort } from './runtime/receiptAcknowledgements';

export const createProductionQueueReceiptAcknowledgements = (store: QueueReceiptStorePort, isActive: () => boolean) =>
  createQueueReceiptAcknowledgements({
    acknowledge: (projectId, queueItemId) => queueBackend.acknowledgeEnqueue!(projectId, queueItemId),
    isActive,
    store,
  });

export const queueCommands: QueueFeatureCommands = {
  cancelCurrentItem: queueBackend.cancelCurrentItem,
  cancelItem: async (itemId) => {
    await queueBackend.cancelItem(itemId);
  },
  cancelScopedItems: queueBackend.cancelScopedItems,
  clearFailedItems: queueBackend.clearFailedItems,
  clearItems: queueBackend.clearItems,
  pauseProcessor: async () => {
    await queueBackend.pauseProcessor();
  },
  resumeProcessor: async () => {
    await queueBackend.resumeProcessor();
  },
};

export const getQueueReadModelOptions = (scope: QueueQueryScope, onRead?: (model: QueueReadModel) => void) =>
  queueReadModelOptions(queueBackend, scope, onRead);

export const createProductionQueueRealtimeRuntime = ({
  coalesceMs,
  invalidate,
  progress,
  refreshModelCache,
}: {
  coalesceMs?: number;
  invalidate: () => void | Promise<void>;
  progress: QueueItemProgressPort;
  refreshModelCache: () => void | Promise<void>;
}): QueueRealtimeRuntime =>
  createQueueRealtimeRuntime({ backend: queueBackend, coalesceMs, invalidate, progress, refreshModelCache });

export const createProductionQueueRuntime = ({
  destinations,
  ensureProjectPersisted,
  ensureTemplatesLoaded,
  history,
  journal,
  locks,
  modelLoads,
  nodeExecution,
  remoteModelTransferToaster,
}: {
  destinations: QueueResultDestinationPort;
  ensureProjectPersisted?(projectId: string): Promise<'ready' | 'refused' | 'retry'>;
  ensureTemplatesLoaded: () => void;
  history: QueueHistoryPort;
  journal?: QueueRunJournalPort;
  locks?: QueueRunLockPort;
  modelLoads: QueueModelLoadPort;
  nodeExecution: QueueNodeExecutionPort;
  remoteModelTransferToaster?: RemoteModelTransferToastPort;
}): QueueRuntime =>
  createQueueRuntime({
    backend: queueBackend,
    destinations,
    ensureProjectPersisted,
    ensureTemplatesLoaded,
    history,
    journal,
    locks,
    modelLoads,
    nodeExecution,
    remoteModelTransferToaster,
  });
