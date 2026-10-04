import { getQueueSummary } from '@features/queue/contracts';
import { useQueueItemProgress } from '@features/queue/react';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';

/** Pair active-project queue summaries with live item progress for consistent chrome surfaces. */
export const useActiveQueueProgress = () => {
  const queueItems = useActiveProjectSelector((project) => project.queue.items);
  const baseSummary = getQueueSummary(queueItems);
  const progress = useQueueItemProgress(baseSummary.runningQueueItemId ?? '');

  return { progress, queueItems, summary: getQueueSummary(queueItems, progress) };
};
