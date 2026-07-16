import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

/**
 * Whether the current user has queue items that are pending or in progress. Drives the invoke
 * button spinners so the user gets feedback that their session is enqueued even when another
 * user's generation occupies the processor. In multiuser mode the global counts include other
 * users' generations, so the per-user counts are preferred when available.
 */
export const useUserHasActiveQueueItems = (): boolean => {
  const { hasActiveQueueItems } = useGetQueueStatusQuery(undefined, {
    selectFromResult: ({ data }) => {
      if (!data) {
        return { hasActiveQueueItems: false };
      }
      const pending = data.queue.user_pending ?? data.queue.pending;
      const inProgress = data.queue.user_in_progress ?? data.queue.in_progress;
      return { hasActiveQueueItems: pending + inProgress > 0 };
    },
  });

  return hasActiveQueueItems;
};
