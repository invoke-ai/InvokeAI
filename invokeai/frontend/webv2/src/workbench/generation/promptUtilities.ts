import { apiFetchJson } from '@workbench/backend/http';

export interface ExpandPromptRequest {
  prompt: string;
  model_key: string;
  max_tokens?: number;
  system_prompt?: string | null;
}

export interface ExpandPromptResponse {
  expanded_prompt: string;
  error?: string | null;
}

export interface ImageToPromptRequest {
  image_name: string;
  model_key: string;
  instruction?: string;
}

export interface ImageToPromptResponse {
  prompt: string;
  error?: string | null;
}

export const expandPrompt = (request: ExpandPromptRequest): Promise<ExpandPromptResponse> =>
  apiFetchJson('/api/v1/utilities/expand-prompt', {
    body: JSON.stringify(request),
    method: 'POST',
  });

export const imageToPrompt = (request: ImageToPromptRequest): Promise<ImageToPromptResponse> =>
  apiFetchJson('/api/v1/utilities/image-to-prompt', {
    body: JSON.stringify(request),
    method: 'POST',
  });
