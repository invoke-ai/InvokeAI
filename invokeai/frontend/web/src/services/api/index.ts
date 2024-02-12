import type { BaseQueryFn, FetchArgs, FetchBaseQueryError, TagDescription } from '@reduxjs/toolkit/query/react';
import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import { $authToken } from 'app/store/nanostores/authToken';
import { $baseUrl } from 'app/store/nanostores/baseUrl';
import { $projectId } from 'app/store/nanostores/projectId';

export const tagTypes = [
  'AppVersion',
  'AppConfig',
  'Board',
  'BoardImagesTotal',
  'BoardAssetsTotal',
  'Image',
  'ImageNameList',
  'ImageList',
  'ImageMetadata',
  'ImageWorkflow',
  'ImageMetadataFromFile',
  'IntermediatesCount',
  'SessionQueueItem',
  'SessionQueueStatus',
  'SessionProcessorStatus',
  'CurrentSessionQueueItem',
  'NextSessionQueueItem',
  'BatchStatus',
  'InvocationCacheStatus',
  'Model',
  'T2IAdapterModel',
  'MainModel',
  'VaeModel',
  'IPAdapterModel',
  'TextualInversionModel',
  'ControlNetModel',
  'LoRAModel',
  'SDXLRefinerModel',
  'Workflow',
  'WorkflowsRecent',
  'Schema',
  // This is invalidated on reconnect. It should be used for queries that have changing data,
  // especially related to the queue and generation.
  'FetchOnReconnect',
] as const;
export type ApiTagDescription = TagDescription<(typeof tagTypes)[number]>;
export const LIST_TAG = 'LIST';

const dynamicBaseQuery: BaseQueryFn<string | FetchArgs, unknown, FetchBaseQueryError> = async (
  args,
  api,
  extraOptions
) => {
  const baseUrl = $baseUrl.get();
  const authToken = $authToken.get();
  const projectId = $projectId.get();

  const rawBaseQuery = fetchBaseQuery({
    baseUrl: baseUrl ? `${baseUrl}/api/v1` : `${window.location.href.replace(/\/$/, '')}/api/v1`,
    prepareHeaders: (headers) => {
      if (authToken) {
        headers.set('Authorization', `Bearer ${authToken}`);
      }
      if (projectId) {
        headers.set('project-id', projectId);
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
