import { useGetProcessorStatusQuery } from 'services/api/endpoints/queue';

export const useIsQueueStarted = () => {
  const { isStarted } = useGetProcessorStatusQuery(undefined, {
    selectFromResult: ({ data }) => {
      if (!data) {
        return { isStarted: false };
      }

      return { isStarted: data.is_started || data.is_processing };
    },
  });

  return isStarted;
};
