import type { QueueFeatureCommands, QueueQueryScope, QueueReadModel } from './core/types';
import type { QueueItemProgressPort, QueueRealtimeRuntime } from './data/realtimeRuntime';
import type { QueueHistoryPort, QueueResultDestinationPort, QueueRuntime } from './runtime';
import type { QueueNodeExecutionPort } from './runtime/coordinator';

import { queueBackend } from './data/httpRealtimeQueueBackend';
import { queueReadModelOptions } from './data/queries';
import { createQueueRealtimeRuntime } from './data/realtimeRuntime';
import { createQueueRuntime } from './runtime';

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
  ensureTemplatesLoaded,
  history,
  nodeExecution,
}: {
  destinations: QueueResultDestinationPort;
  ensureTemplatesLoaded: () => void;
  history: QueueHistoryPort;
  nodeExecution: QueueNodeExecutionPort;
}): QueueRuntime =>
  createQueueRuntime({ backend: queueBackend, destinations, ensureTemplatesLoaded, history, nodeExecution });
