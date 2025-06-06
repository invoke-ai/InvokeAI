import { useGetBatchStatusQuery } from 'services/api/endpoints/queue';

export const useBatchIsCanceled = (batch_id: string) => {
  const { isCanceled } = useGetBatchStatusQuery(
    { batch_id },
    {
      selectFromResult: ({ data }) => {
        if (!data) {
          return { isCanceled: true };
        }

        return {
          isCanceled: data?.in_progress === 0 && data?.pending === 0,
        };
      },
    }
  );

  return isCanceled;
};
