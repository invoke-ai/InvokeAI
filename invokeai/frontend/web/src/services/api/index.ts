import { createSelectorCreator, lruMemoize } from '@reduxjs/toolkit';
import type {
  BaseQueryFn,
  FetchArgs,
  FetchBaseQueryArgs,
  FetchBaseQueryError,
  TagDescription,
} from '@reduxjs/toolkit/query/react';
import { buildCreateApi, coreModule, fetchBaseQuery, reactHooksModule } from '@reduxjs/toolkit/query/react';
import queryString from 'query-string';
import stableHash from 'stable-hash';

const tagTypes = [
  'AppVersion',
  'AppConfig',
  'Board',
  'BoardImagesTotal',
  'BoardAssetsTotal',
  'HFTokenStatus',
  'Image',
  'ImageNameList',
  'ImageList',
  'ImageMetadata',
  'ImageWorkflow',
  'ImageCollectionCounts',
  'ImageCollection',
  'ImageMetadataFromFile',
  'IntermediatesCount',
  'SessionQueueItemIdList',
  'SessionQueueItem',
  'SessionQueueStatus',
  'SessionProcessorStatus',
  'CurrentSessionQueueItem',
  'NextSessionQueueItem',
  'BatchStatus',
  'InvocationCacheStatus',
  'ModelConfig',
  'ModelInstalls',
  'ModelRelationships',
  'ModelScanFolderResults',
  'T2IAdapterModel',
  'MainModel',
  'VaeModel',
  'IPAdapterModel',
  'TextualInversionModel',
  'ControlNetModel',
  'LoRAModel',
  'SDXLRefinerModel',
  'Workflow',
  'WorkflowTagCounts',
  'WorkflowCategoryCounts',
  'StylePreset',
  'Schema',
  'QueueCountsByDestination',
  // This is invalidated on reconnect. It should be used for queries that have changing data,
  // especially related to the queue and generation.
  'FetchOnReconnect',
  'ClientState',
] as const;
export type ApiTagDescription = TagDescription<(typeof tagTypes)[number]>;
export const LIST_TAG = 'LIST';
export const LIST_ALL_TAG = 'LIST_ALL';

export const getBaseUrl = (): string => {
  return window.location.href.replace(/\/$/, '');
};

const dynamicBaseQuery: BaseQueryFn<string | FetchArgs, unknown, FetchBaseQueryError> = (args, api, extraOptions) => {
  const isOpenAPIRequest =
    (args instanceof Object && args.url.includes('openapi.json')) ||
    (typeof args === 'string' && args.includes('openapi.json'));

  const fetchBaseQueryArgs: FetchBaseQueryArgs = {
    baseUrl: getBaseUrl(),
  };

  // When fetching the openapi.json, we need to remove circular references from the JSON.
  if (isOpenAPIRequest) {
    fetchBaseQueryArgs.jsonReplacer = getCircularReplacer();
  }

  const rawBaseQuery = fetchBaseQuery(fetchBaseQueryArgs);

  return rawBaseQuery(args, api, extraOptions);
};

const createLruSelector = createSelectorCreator({
  memoize: lruMemoize,
  argsMemoize: lruMemoize,
});

const customCreateApi = buildCreateApi(
  coreModule({ createSelector: createLruSelector }),
  reactHooksModule({ createSelector: createLruSelector })
);

export const api = customCreateApi({
  baseQuery: dynamicBaseQuery,
  reducerPath: 'api',
  tagTypes,
  endpoints: () => ({}),
  invalidationBehavior: 'immediately',
  serializeQueryArgs: stableHash,
  refetchOnReconnect: true,
});

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

export const buildV1Url = (path: string, query?: Parameters<typeof queryString.stringify>[0]): string => {
  if (!query) {
    return `api/v1/${path}`;
  }
  return `api/v1/${path}?${queryString.stringify(query)}`;
};
export const buildV2Url = (path: string): string => `api/v2/${path}`;
