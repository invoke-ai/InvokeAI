import type { paths } from 'services/api/schema';

import { api, buildV1Url } from '..';

/**
 * Builds an endpoint URL for the utilities router
 * @example
 * buildUtilitiesUrl('some-path')
 * // '/api/v1/utilities/some-path'
 */
const buildUtilitiesUrl = (path: string = '') => buildV1Url(`utilities/${path}`);

export const utilitiesApi = api.injectEndpoints({
  endpoints: (build) => ({
    dynamicPrompts: build.query<
      paths['/api/v1/utilities/dynamicprompts']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/utilities/dynamicprompts']['post']['requestBody']['content']['application/json']
    >({
      query: (arg) => ({
        url: buildUtilitiesUrl('dynamicprompts'),
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
