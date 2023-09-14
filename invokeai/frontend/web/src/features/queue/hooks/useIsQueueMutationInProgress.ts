import {
  useCancelByBatchIdsMutation,
  useCancelQueueExecutionMutation,
  useClearQueueMutation,
  useEnqueueBatchMutation,
  useEnqueueGraphMutation,
  usePruneQueueMutation,
  useStartQueueExecutionMutation,
  useStopQueueExecutionMutation,
} from 'services/api/endpoints/queue';

export const useIsQueueMutationInProgress = () => {
  const [_triggerEnqueueBatch, { isLoading: isLoadingEnqueueBatch }] =
    useEnqueueBatchMutation({
      fixedCacheKey: 'enqueueBatch',
    });
  const [_triggerEnqueueGraph, { isLoading: isLoadingEnqueueGraph }] =
    useEnqueueGraphMutation({
      fixedCacheKey: 'enqueueGraph',
    });
  const [_triggerStartQueue, { isLoading: isLoadingStartQueue }] =
    useStartQueueExecutionMutation({
      fixedCacheKey: 'startQueue',
    });
  const [_triggerStopQueue, { isLoading: isLoadingStopQueue }] =
    useStopQueueExecutionMutation({
      fixedCacheKey: 'stopQueue',
    });
  const [_triggerCancelQueue, { isLoading: isLoadingCancelQueue }] =
    useCancelQueueExecutionMutation({
      fixedCacheKey: 'cancelQueue',
    });
  const [_triggerClearQueue, { isLoading: isLoadingClearQueue }] =
    useClearQueueMutation({
      fixedCacheKey: 'clearQueue',
    });
  const [_triggerPruneQueue, { isLoading: isLoadingPruneQueue }] =
    usePruneQueueMutation({
      fixedCacheKey: 'pruneQueue',
    });
  const [_triggerCancelByBatchIds, { isLoading: isLoadingCancelByBatchIds }] =
    useCancelByBatchIdsMutation({
      fixedCacheKey: 'cancelByBatchIds',
    });
  return (
    isLoadingEnqueueBatch ||
    isLoadingEnqueueGraph ||
    isLoadingStartQueue ||
    isLoadingStopQueue ||
    isLoadingCancelQueue ||
    isLoadingClearQueue ||
    isLoadingPruneQueue ||
    isLoadingCancelByBatchIds
  );
};
