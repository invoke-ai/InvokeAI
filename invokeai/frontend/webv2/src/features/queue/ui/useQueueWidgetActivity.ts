import { getQueueActivity, type QueueActivity } from '@features/queue/core/queueActivity';
import { getPersonalQueueActivity } from '@features/queue/core/types';

import { useIsProcessorPaused, useQueueCounts } from './queueDataStore';
import { useLoadingModelLabel, useQueueUi } from './QueueUiContext';

/**
 * The queue widget's reading of the shared activity model, from the scope's counts of the viewer's own work: the same
 * states, in the same precedence, as the top bar and status chip read from the project's runs.
 */
export const useQueueWidgetActivity = (): {
  activity: QueueActivity;
  inProgress: number;
  loadingModel: string | null;
  pending: number;
} => {
  const { isConnected } = useQueueUi();
  const { inProgress, pending } = getPersonalQueueActivity(useQueueCounts());
  const isPaused = useIsProcessorPaused();
  const loadingModel = useLoadingModelLabel();

  return {
    activity: getQueueActivity({
      hasOpenWork: inProgress + pending > 0,
      isConnected,
      isLoadingModels: loadingModel !== null,
      isPaused,
    }),
    inProgress,
    loadingModel,
    pending,
  };
};
