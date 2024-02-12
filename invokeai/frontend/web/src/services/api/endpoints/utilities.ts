import { $openAPISchemaUrl } from 'app/store/nanostores/openAPISchemaUrl';
import type { components } from 'services/api/schema';

import { api } from '..';

function getCircularReplacer() {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const ancestors: Record<string, any>[] = [];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return function (key: string, value: any) {
    if (typeof value !== 'object' || value === null) {
      return value;
    }
    // `this` is the object that value is contained in, i.e., its direct parent.
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore don't think it's possible to not have TS complain about this...
    while (ancestors.length > 0 && ancestors.at(-1) !== this) {
      ancestors.pop();
    }
    if (ancestors.includes(value)) {
      return '[Circular]';
    }
    ancestors.push(value);
    return value;
  };
}

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
    loadSchema: build.query({
      queryFn: async () => {
        try {
          const openAPISchemaUrl = $openAPISchemaUrl.get();

          const url = openAPISchemaUrl ? openAPISchemaUrl : `${window.location.href.replace(/\/$/, '')}/openapi.json`;
          const response = await fetch(url);
          const openAPISchema = await response.json();

          const schemaJSON = JSON.parse(JSON.stringify(openAPISchema, getCircularReplacer()));

          return { data: schemaJSON };
        } catch (error) {
          console.error({ error });
          return {
            error: {
              status: 500,
              statusText: 'Internal Server Error',
              data: 'Could not load openAPI schema',
            },
          };
        }
      },
      providesTags: ['Schema'],
    }),
  }),
});

export const { useDynamicPromptsQuery, useLoadSchemaQuery, useLazyLoadSchemaQuery } = utilitiesApi;
