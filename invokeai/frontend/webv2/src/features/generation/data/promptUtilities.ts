import { apiFetchJson } from '@platform/transport/http';

export interface ExpandPromptRequest {
  prompt: string;
  model_key: string;
  task_id: string;
  max_tokens?: number;
  seed?: number | null;
  system_prompt?: string | null;
  /** Needs a model whose record has `supports_images`. */
  image_name?: string;
}

export interface ExpandPromptResponse {
  expanded_prompt: string;
  seed: number;
  error?: string | null;
}

export interface ImageToPromptRequest {
  image_name: string;
  model_key: string;
  task_id: string;
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

export interface ParseDynamicPromptsRequest {
  prompt: string;
  max_prompts?: number;
  combinatorial?: boolean;
  /** Only read by the random generator; ignored when combinatorial. */
  seed?: number | null;
}

export interface ParseDynamicPromptsResponse {
  prompts: string[];
  /** Soft expansion errors can still return usable prompts. */
  error?: string | null;
}

export const parseDynamicPrompts = (
  request: ParseDynamicPromptsRequest,
  signal?: AbortSignal
): Promise<ParseDynamicPromptsResponse> =>
  apiFetchJson('/api/v1/utilities/dynamicprompts', {
    body: JSON.stringify(request),
    method: 'POST',
    signal,
  });
