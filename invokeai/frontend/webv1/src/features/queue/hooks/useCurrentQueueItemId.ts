import { useGetCurrentQueueItemQuery } from 'services/api/endpoints/queue';

export const useCurrentQueueItemId = () => {
  const { currentQueueItemId } = useGetCurrentQueueItemQuery(undefined, {
    selectFromResult: ({ data }) => ({
      currentQueueItemId: data?.item_id ?? null,
    }),
  });

  return currentQueueItemId;
};
