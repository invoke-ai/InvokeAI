import type {
  CanvasStagingCandidateContract,
  CommitStagedImageOptions,
  CommitStagedImageResult,
} from '@workbench/canvas-engine/api';
import type { WorkbenchQueueItem as QueueItem } from '@workbench/queueHistoryContracts';

import { isCancellableQueueItem } from '@workbench/canvasStagingView';

/**
 * The local queue item (one backend batch) that produced `candidate`, while it still has queued or running work the
 * queue can cancel; `null` once it has finished or cannot be cancelled.
 */
export const getStoppableCandidateBatch = (
  candidate: CanvasStagingCandidateContract,
  queueItems: readonly QueueItem[]
): string | null => {
  const item = queueItems.find((queueItem) => queueItem.id === candidate.sourceQueueItemId);
  return item && isCancellableQueueItem(item) ? item.id : null;
};

export interface StagedAcceptancePorts {
  commit(options: CommitStagedImageOptions): CommitStagedImageResult;
  /** Requests cancellation through the queue runtime, which owns retries and late-result fencing. */
  stopBatch(queueItemId: string): void;
}

/**
 * Accepts the candidate, then stops the rest of its batch so later siblings cannot reopen staging. The stop is
 * requested only after the layer has landed, so a refused accept leaves generation running; banking a disabled
 * layer keeps comparing, and the batch with it.
 */
export const acceptStagedCandidate = (
  ports: StagedAcceptancePorts,
  options: CommitStagedImageOptions & { queueItems: readonly QueueItem[] }
): CommitStagedImageResult => {
  const { queueItems, ...commitOptions } = options;
  const batch = commitOptions.continueStaging ? null : getStoppableCandidateBatch(commitOptions.candidate, queueItems);
  const result = ports.commit(commitOptions);
  if (result.status === 'committed' && batch) {
    ports.stopBatch(batch);
  }
  return result;
};
