import type { QueueItem } from '@features/queue/core/historyTypes';
import type { QueueSourceId } from '@features/queue/core/types';

/** Source kinds whose pending queue items the runtime enqueues to the backend. */
export const BACKEND_SUBMITTABLE_SOURCE_IDS = [
  'generate',
  'workflow',
  'upscale',
  'canvas',
] as const satisfies readonly QueueSourceId[];

const BACKEND_SUBMITTABLE_SOURCE_ID_SET: ReadonlySet<QueueSourceId> = new Set(BACKEND_SUBMITTABLE_SOURCE_IDS);

/** Whether a queue item's source is one the runtime knows how to enqueue. */
export const isBackendSubmittableSourceId = (sourceId: QueueSourceId): boolean =>
  BACKEND_SUBMITTABLE_SOURCE_ID_SET.has(sourceId);

/**
 * Whether a queue item is a fresh, backend-submittable `pending` item — the
 * runtime still applies its own "already started this id" dedupe on top of this.
 */
export const shouldSubmitPendingQueueItem = (queueItem: QueueItem): boolean =>
  queueItem.status === 'pending' && isBackendSubmittableSourceId(queueItem.snapshot.sourceId);
