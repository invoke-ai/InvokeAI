import { getUserScopedQueueCounts } from 'features/queue/store/userScopedQueueCounts';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

/**
 * Whether the current user has queue items that are pending or in progress. Drives the invoke
 * button spinners so the user gets feedback that their session is enqueued even when another
 * user's generation occupies the processor.
 */
export const useUserHasActiveQueueItems = (): boolean => {
  const { hasActiveQueueItems } = useGetQueueStatusQuery(undefined, {
    selectFromResult: ({ data }) => {
      if (!data) {
        return { hasActiveQueueItems: false };
      }
      const { pending, inProgress } = getUserScopedQueueCounts(data.queue);
      return { hasActiveQueueItems: pending + inProgress > 0 };
    },
  });

  return hasActiveQueueItems;
};
