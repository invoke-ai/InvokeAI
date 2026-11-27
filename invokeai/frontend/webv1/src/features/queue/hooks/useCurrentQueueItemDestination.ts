import { useGetCurrentQueueItemQuery } from 'services/api/endpoints/queue';

export const useCurrentQueueItemDestination = () => {
  const { currentQueueItemDestination } = useGetCurrentQueueItemQuery(undefined, {
    selectFromResult: ({ data }) => ({
      currentQueueItemDestination: data?.destination ?? null,
    }),
  });

  return currentQueueItemDestination;
};
