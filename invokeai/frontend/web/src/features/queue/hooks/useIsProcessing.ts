import { useMemo } from 'react';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

export const useIsProcessing = () => {
  const { currentData } = useGetQueueStatusQuery();
  const isProcessing = useMemo(() => {
    if ((currentData?.processor.is_processing && currentData?.queue.pending) || currentData?.queue.in_progress) {
      return true;
    }
    return false;
  }, [currentData?.processor.is_processing, currentData?.queue.in_progress, currentData?.queue.pending]);
  return isProcessing;
};
