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

/**
 * The longest failure message a submission retry keeps. A 5xx without a JSON body is the whole response (an HTML page,
 * a traceback), and the retry is written to state and recovery storage on every attempt. The final failure shows the
 * message as the run's error, so it keeps the leading text rather than one line.
 */
const MAX_SUBMISSION_RETRY_MESSAGE_LENGTH = 500;

/**
 * The message cut to MAX_SUBMISSION_RETRY_MESSAGE_LENGTH, ending with an ellipsis when cut. The cut never keeps half
 * of a surrogate pair.
 */
export const boundSubmissionRetryMessage = (message: string): string => {
  if (message.length <= MAX_SUBMISSION_RETRY_MESSAGE_LENGTH) {
    return message;
  }
  const kept = message.slice(0, MAX_SUBMISSION_RETRY_MESSAGE_LENGTH - 1);
  return `${/[\uD800-\uDBFF]$/.test(kept) ? kept.slice(0, -1) : kept}…`;
};
