import { useMemo } from 'react';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

export const useIsProcessing = () => {
  const { data } = useGetQueueStatusQuery();
  const isProcessing = useMemo(() => {
    if ((data?.processor.is_processing && data?.queue.pending) || data?.queue.in_progress) {
      return true;
    }
    return false;
  }, [data?.processor.is_processing, data?.queue.in_progress, data?.queue.pending]);
  return isProcessing;
};
