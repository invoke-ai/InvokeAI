import { useGetCurrentQueueItemQuery } from 'services/api/endpoints/queue';

export const useCurrentDestination = () => {
  const { destination } = useGetCurrentQueueItemQuery(undefined, {
    selectFromResult: ({ data }) => ({
      destination: data ? data.destination : null,
    }),
  });

  return destination;
};
