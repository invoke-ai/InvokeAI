import { useStore } from '@nanostores/react';
import { $promptExpansionRequest } from 'services/events/stores';

export const usePromptExpansionTracking = () => {
  const request = useStore($promptExpansionRequest);

  const isPending = request?.status === 'pending';
  const isCompleted = request?.status === 'completed';
  const isFailed = request?.status === 'error';
  const hasRequest = request !== null;

  const getDuration = (): number | undefined => {
    if (!request || request.status === 'pending') {
      return undefined;
    }
    return Date.now() - request.startTime;
  };

  return {
    request,
    isPending,
    isCompleted,
    isFailed,
    hasRequest,
    getDuration,
  };
};
