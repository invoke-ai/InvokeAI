import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

export const useIsQueueEmpty = () => {
  const { isEmpty } = useGetQueueStatusQuery(undefined, {
    selectFromResult: ({ data }) => {
      if (!data) {
        return { isEmpty: true };
      }
      return { isEmpty: data.in_progress === 0 && data.pending === 0 };
    },
  });
  return isEmpty;
};
