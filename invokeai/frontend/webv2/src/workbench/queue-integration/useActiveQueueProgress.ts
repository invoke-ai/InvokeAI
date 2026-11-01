import { useModelLoads } from '@features/models';
import { getQueueActivity, getQueueSummary } from '@features/queue/contracts';
import { useIsProcessorPaused, useQueueItemProgress, useQueueOpenWorkElsewhere } from '@features/queue/react';
import { useActiveProjectSelector, useWorkbenchSelector } from '@workbench/WorkbenchContext';

/** Pair active-project queue summaries with live item progress for consistent chrome surfaces. */
export const useActiveQueueProgress = () => {
  const queueItems = useActiveProjectSelector((project) => project.queue.items);
  const baseSummary = getQueueSummary(queueItems);
  const progress = useQueueItemProgress(baseSummary.runningQueueItemId ?? '');

  return { progress, queueItems, summary: getQueueSummary(queueItems, progress) };
};

/**
 * The top bar's and status chip's reading of the queue. Progress and the image countdown come from the project's runs;
 * the count of open work adds the viewer's open work elsewhere in the queue scope (other projects, other tabs), which
 * the project's runs never include, so the two surfaces count what the queue widget counts under the same scope
 * setting without counting the project's own items twice.
 */
export const useQueueChromeActivity = () => {
  const local = useActiveQueueProgress();
  const elsewhere = useQueueOpenWorkElsewhere(local.queueItems);
  const modelLoads = useModelLoads();
  const isPaused = useIsProcessorPaused();
  const isConnected = useWorkbenchSelector((snapshot) => snapshot.backendConnection.status === 'connected');
  const remaining = local.summary.remaining + elsewhere;

  return {
    ...local,
    activity: getQueueActivity({
      hasOpenWork: remaining > 0,
      isConnected,
      isLoadingModels: modelLoads.length > 0,
      isPaused,
    }),
    elsewhere,
    modelLoads,
    remaining,
  };
};
