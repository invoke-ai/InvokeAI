import type { paths } from 'services/api/schema';

import { api, buildV1Url } from '..';

/**
 * Builds an endpoint URL for the utilities router
 * @example
 * buildUtilitiesUrl('some-path')
 * // '/api/v1/utilities/some-path'
 */
const buildUtilitiesUrl = (path: string = '') => buildV1Url(`utilities/${path}`);

// Types for expand-prompt and image-to-prompt endpoints.
// These will use generated schema types once schema.ts is regenerated.
type ExpandPromptRequest = {
  prompt: string;
  model_key: string;
  max_tokens?: number;
  system_prompt?: string | null;
};

type ExpandPromptResponse = {
  expanded_prompt: string;
  error?: string | null;
};

type ImageToPromptRequest = {
  image_name: string;
  model_key: string;
  instruction?: string;
};

type ImageToPromptResponse = {
  prompt: string;
  error?: string | null;
};

export type WildcardsResponse =
  paths['/api/v1/utilities/wildcards']['get']['responses']['200']['content']['application/json'];
export type WildcardIndexItem = WildcardsResponse['wildcards'][number];
export type WildcardValuesResponse =
  paths['/api/v1/utilities/wildcards/values']['get']['responses']['200']['content']['application/json'];

export const utilitiesApi = api.injectEndpoints({
  endpoints: (build) => ({
    wildcards: build.query<WildcardsResponse, void>({
      query: () => ({
        url: buildUtilitiesUrl('wildcards'),
        method: 'GET',
      }),
      providesTags: ['FetchOnReconnect'],
    }),
    wildcardValues: build.query<WildcardValuesResponse, { path: string; limit?: number }>({
      query: ({ path, limit = 200 }) => ({
        url: buildUtilitiesUrl('wildcards/values'),
        method: 'GET',
        params: { path, limit },
      }),
      providesTags: ['FetchOnReconnect'],
    }),
    dynamicPrompts: build.query<
      paths['/api/v1/utilities/dynamicprompts']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/utilities/dynamicprompts']['post']['requestBody']['content']['application/json']
    >({
      query: (arg) => ({
        url: buildUtilitiesUrl('dynamicprompts'),
        body: arg,
        method: 'POST',
      }),
      // We need to fetch this on reconnect bc the user may have changed the text field while
      // disconnected.
      providesTags: ['FetchOnReconnect'],
    }),
    expandPrompt: build.mutation<ExpandPromptResponse, ExpandPromptRequest>({
      query: (arg) => ({
        url: buildUtilitiesUrl('expand-prompt'),
        body: arg,
        method: 'POST',
      }),
    }),
    imageToPrompt: build.mutation<ImageToPromptResponse, ImageToPromptRequest>({
      query: (arg) => ({
        url: buildUtilitiesUrl('image-to-prompt'),
        body: arg,
        method: 'POST',
      }),
    }),
  }),
});

export const { useExpandPromptMutation, useImageToPromptMutation, useLazyWildcardValuesQuery, useWildcardsQuery } =
  utilitiesApi;
