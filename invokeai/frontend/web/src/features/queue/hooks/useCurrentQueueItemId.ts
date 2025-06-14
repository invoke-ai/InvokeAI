import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

export const useCurrentQueueItemId = () => {
  const { currentQueueItemId } = useGetQueueStatusQuery(undefined, {
    selectFromResult: ({ data }) => ({
      currentQueueItemId: data?.queue.item_id ?? null,
    }),
  });

  return currentQueueItemId;
};
