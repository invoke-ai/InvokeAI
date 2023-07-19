import { FullTagDescription } from '@reduxjs/toolkit/dist/query/endpointDefinitions';
import {
  BaseQueryFn,
  FetchArgs,
  FetchBaseQueryError,
  createApi,
  fetchBaseQuery,
} from '@reduxjs/toolkit/query/react';
import { $authToken, $baseUrl } from 'services/api/client';

export const tagTypes = [
  'Board',
  'Image',
  'ImageNameList',
  'ImageList',
  'ImageMetadata',
  'Model',
];
export type ApiFullTagDescription = FullTagDescription<
  (typeof tagTypes)[number]
>;
export const LIST_TAG = 'LIST';

const dynamicBaseQuery: BaseQueryFn<
  string | FetchArgs,
  unknown,
  FetchBaseQueryError
> = async (args, api, extraOptions) => {
  const baseUrl = $baseUrl.get();
  const authToken = $authToken.get();

  const rawBaseQuery = fetchBaseQuery({
    baseUrl: `${baseUrl ?? ''}/api/v1`,
    prepareHeaders: (headers) => {
      if (authToken) {
        headers.set('Authorization', `Bearer ${authToken}`);
      }

      return headers;
    },
  });

  return rawBaseQuery(args, api, extraOptions);
};

export const api = createApi({
  baseQuery: dynamicBaseQuery,
  reducerPath: 'api',
  tagTypes,
  endpoints: () => ({}),
});
