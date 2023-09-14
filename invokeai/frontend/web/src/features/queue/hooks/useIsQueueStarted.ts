import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

export const useIsQueueStarted = () => {
  const { isStarted } = useGetQueueStatusQuery(undefined, {
    selectFromResult: ({ data }) => {
      if (!data) {
        return { isStarted: false };
      }

      return { isStarted: data.started || data.stop_after_current };
    },
  });

  return isStarted;
};
