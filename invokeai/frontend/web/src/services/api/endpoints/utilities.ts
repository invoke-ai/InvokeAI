import type { components } from 'services/api/schema';

import { api } from '..';

export const utilitiesApi = api.injectEndpoints({
  endpoints: (build) => ({
    dynamicPrompts: build.query<
      components['schemas']['DynamicPromptsResponse'],
      { prompt: string; max_prompts: number }
    >({
      query: (arg) => ({
        url: 'utilities/dynamicprompts',
        body: arg,
        method: 'POST',
      }),
      keepUnusedDataFor: 86400, // 24 hours
      // We need to fetch this on reconnect bc the user may have changed the text field while
      // disconnected.
      providesTags: ['FetchOnReconnect'],
    }),
  }),
});

export const { useDynamicPromptsQuery } = utilitiesApi;
