import type { ImageDTO } from 'services/api/types';
import { $promptExpansionRequest } from 'services/events/stores';

export interface PromptExpansionRequest {
  startTime: number;
  status: 'pending' | 'completed' | 'error';
  sourceImage?: ImageDTO;
}

export const getPromptExpansionRequest = () => {
  return $promptExpansionRequest.get();
};

export const isPromptExpansionPending = () => {
  const request = $promptExpansionRequest.get();
  return request?.status === 'pending';
};

export const isPromptExpansionCompleted = () => {
  const request = $promptExpansionRequest.get();
  return request?.status === 'completed';
};

export const isPromptExpansionFailed = () => {
  const request = $promptExpansionRequest.get();
  return request?.status === 'error';
};

export const clearPromptExpansionRequest = () => {
  $promptExpansionRequest.set(null);
};

export const getPromptExpansionDuration = () => {
  const request = $promptExpansionRequest.get();
  if (!request || request.status === 'pending') {
    return undefined;
  }
  return Date.now() - request.startTime;
};
