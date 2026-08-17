import { getQueueSummary } from '@features/queue/contracts';
import { useQueueItemProgress } from '@features/queue/react';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';

/**
 * The one pairing every aggregate progress surface needs: the active project's
 * queue summary plus the running item's live progress. Extracted because four
 * chrome surfaces (Invoke button, tab title, queue cluster, queue-status chip)
 * had copies that could — and did — drift on edge-case handling.
 */
export const useActiveQueueProgress = () => {
  const queueItems = useActiveProjectSelector((project) => project.queue.items);
  const baseSummary = getQueueSummary(queueItems);
  const progress = useQueueItemProgress(baseSummary.runningQueueItemId ?? '');

  return { progress, queueItems, summary: getQueueSummary(queueItems, progress) };
};
