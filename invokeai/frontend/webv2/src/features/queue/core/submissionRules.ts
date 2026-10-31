import type { QueueItem } from '@features/queue/core/historyTypes';
import type { QueueSourceId } from '@features/queue/core/types';

/** The graph nodes whose outputs are a source's results; Workflow runs deliver every image they produce. */
export const QUEUE_SOURCE_RESULT_NODE_IDS: Readonly<Partial<Record<QueueSourceId, readonly string[]>>> = {
  canvas: ['canvas_output'],
  generate: ['canvas_output'],
  upscale: ['upscale_output'],
  video: ['video_output'],
};

/** The snapshot fields that limit a source's queued run to its result nodes. */
export const getQueueSourceResultSelection = (sourceId: QueueSourceId): { resultNodeIds?: string[] } => {
  const resultNodeIds = QUEUE_SOURCE_RESULT_NODE_IDS[sourceId];

  return resultNodeIds ? { resultNodeIds: [...resultNodeIds] } : {};
};

/** Source kinds whose pending queue items the runtime enqueues to the backend. */
export const BACKEND_SUBMITTABLE_SOURCE_IDS = [
  'generate',
  'workflow',
  'upscale',
  'video',
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
