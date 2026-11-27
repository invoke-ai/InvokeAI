import type {
  CanvasStagingCandidateContract,
  CommitStagedImageOptions,
  CommitStagedImageResult,
} from '@workbench/canvas-engine/api';
import type { WorkbenchQueueItem as QueueItem } from '@workbench/queueHistoryContracts';

import { isCancellableQueueItem } from '@workbench/canvasStagingView';

/** The candidate's local queue item while it still has cancellable work; `null` once finished. */
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

/** Accepts, then stops the rest of the batch; the stop waits for the layer so a refused accept keeps running. */
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
