import {
  useCancelQueueItemMutation,
  // useCancelByBatchIdsMutation,
  useClearQueueMutation,
  useEnqueueBatchMutation,
  usePauseProcessorMutation,
  usePruneQueueMutation,
  useResumeProcessorMutation,
} from 'services/api/endpoints/queue';

export const useIsQueueMutationInProgress = () => {
  const [_triggerEnqueueBatch, { isLoading: isLoadingEnqueueBatch }] = useEnqueueBatchMutation({
    fixedCacheKey: 'enqueueBatch',
  });
  const [_triggerResumeProcessor, { isLoading: isLoadingResumeProcessor }] = useResumeProcessorMutation({
    fixedCacheKey: 'resumeProcessor',
  });
  const [_triggerPauseProcessor, { isLoading: isLoadingPauseProcessor }] = usePauseProcessorMutation({
    fixedCacheKey: 'pauseProcessor',
  });
  const [_triggerCancelQueue, { isLoading: isLoadingCancelQueue }] = useCancelQueueItemMutation({
    fixedCacheKey: 'cancelQueueItem',
  });
  const [_triggerClearQueue, { isLoading: isLoadingClearQueue }] = useClearQueueMutation({
    fixedCacheKey: 'clearQueue',
  });
  const [_triggerPruneQueue, { isLoading: isLoadingPruneQueue }] = usePruneQueueMutation({
    fixedCacheKey: 'pruneQueue',
  });
  // const [_triggerCancelByBatchIds, { isLoading: isLoadingCancelByBatchIds }] =
  //   useCancelByBatchIdsMutation({
  //     fixedCacheKey: 'cancelByBatchIds',
  //   });
  return (
    isLoadingEnqueueBatch ||
    isLoadingResumeProcessor ||
    isLoadingPauseProcessor ||
    isLoadingCancelQueue ||
    isLoadingClearQueue ||
    isLoadingPruneQueue
    // isLoadingCancelByBatchIds
  );
};
