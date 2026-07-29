import { createSelectorCreator, lruMemoize } from '@reduxjs/toolkit';
import type {
  BaseQueryFn,
  FetchArgs,
  FetchBaseQueryArgs,
  FetchBaseQueryError,
  TagDescription,
} from '@reduxjs/toolkit/query/react';
import { buildCreateApi, coreModule, fetchBaseQuery, reactHooksModule } from '@reduxjs/toolkit/query/react';
import { getDeploymentBaseUrl } from 'common/util/baseUrl';
import { sessionExpiredLogout, tokenRefreshed } from 'features/auth/store/authSlice';
import {
  beginAuthTransition,
  captureAuthGeneration,
  isTokenRefreshThrottled,
  markTokenRefreshAccepted,
  MEDIA_COOKIE_SYNC_TIMEOUT_MS,
  runWithMediaAuthLock,
  shouldAcceptRefreshedToken,
} from 'features/auth/store/authTokenRefresh';
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
  'ImageMoveStatus',
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
  'OrphanedModels',
  'T2IAdapterModel',
  'MainModel',
  'VaeModel',
  'IPAdapterModel',
  'TextualInversionModel',
  'ControlNetModel',
  'LoRAModel',
  'SDXLRefinerModel',
  'Workflow',
  'WorkflowTags',
  'WorkflowTagCounts',
  'WorkflowCategoryCounts',
  'StylePreset',
  'Schema',
  'QueueCountsByDestination',
  // This is invalidated on reconnect. It should be used for queries that have changing data,
  // especially related to the queue and generation.
  'FetchOnReconnect',
  'ClientState',
  'UserList',
  'CustomNodePacks',
  'VirtualBoards',
  // Video tags (parallel to Image tags).
  'Video',
  'VideoList',
  'VideoMetadata',
  'VideoWorkflow',
  'VideoNameList',
  'BoardVideosTotal',
  // Polymorphic gallery list (images + videos interleaved by created_at).
  'GalleryItemList',
  'GalleryItemNameList',
] as const;
export type ApiTagDescription = TagDescription<(typeof tagTypes)[number]>;
export const LIST_TAG = 'LIST';
export const LIST_ALL_TAG = 'LIST_ALL';

export const getBaseUrl = (): string => {
  return getDeploymentBaseUrl();
};

const dynamicBaseQuery: BaseQueryFn<string | FetchArgs, unknown, FetchBaseQueryError> = async (
  args,
  api,
  extraOptions
) => {
  const isOpenAPIRequest =
    (args instanceof Object && args.url.includes('openapi.json')) ||
    (typeof args === 'string' && args.includes('openapi.json'));

  const isAuthEndpoint =
    (args instanceof Object &&
      typeof args.url === 'string' &&
      (args.url.includes('/auth/login') || args.url.includes('/auth/setup'))) ||
    (typeof args === 'string' && (args.includes('/auth/login') || args.includes('/auth/setup')));

  const token = localStorage.getItem('auth_token');
  const requestUrl = typeof args === 'string' ? args : args.url;
  const isAuthTransition = requestUrl.includes('/auth/login') || requestUrl.includes('/auth/logout');
  if (isAuthTransition) {
    beginAuthTransition();
  }
  const requestGeneration = captureAuthGeneration();
  const changesMediaCookie = isAuthTransition || requestUrl.includes('/auth/media-cookie');

  const fetchBaseQueryArgs: FetchBaseQueryArgs = {
    baseUrl: getBaseUrl(),
    prepareHeaders: (headers) => {
      // Add auth token to all requests except setup and login
      if (token && !isAuthEndpoint) {
        headers.set('Authorization', `Bearer ${token}`);
      }
      return headers;
    },
  };

  // When fetching the openapi.json, we need to remove circular references from the JSON.
  if (isOpenAPIRequest) {
    fetchBaseQueryArgs.jsonReplacer = getCircularReplacer();
  }

  const rawBaseQuery = fetchBaseQuery(fetchBaseQueryArgs);

  const execute = () => rawBaseQuery(args, api, extraOptions);
  const result = changesMediaCookie ? await runWithMediaAuthLock(execute) : await execute();

  // If we sent an auth token but got 401, the token is invalid/expired.
  // Only trigger session expiry when we actually sent a token — unauthenticated
  // requests (e.g. client_state queries during page load) should not cause logout.
  if (result.error && result.error.status === 401 && !isAuthEndpoint && token) {
    api.dispatch(sessionExpiredLogout());
  }

  // Sliding window token refresh: if the server returned a refreshed token,
  // update localStorage so subsequent requests use the new expiry.
  if (!result.error && result.meta?.response) {
    const refreshedToken = result.meta.response.headers.get('X-Refreshed-Token');
    if (refreshedToken && token) {
      await acceptRefreshedToken(refreshedToken, token, requestGeneration, api.dispatch);
    }
  }

  return result;
};

/**
 * Accepts a refreshed bearer token minted by the sliding-window middleware: syncs the
 * media cookie under the cross-tab lock, then commits the token. Shared by
 * dynamicBaseQuery and the client-state persistence driver — the driver bypasses RTK
 * Query with raw fetch, and without this its POSTs would discard every refreshed
 * token, so a persistence-only session (e.g. canvas tweaking all day) would hard-expire
 * despite constant activity.
 */
export const acceptRefreshedToken = async (
  refreshedToken: string,
  requestToken: string,
  requestGeneration: number,
  dispatch: (action: ReturnType<typeof tokenRefreshed>) => unknown
): Promise<void> => {
  if (isTokenRefreshThrottled() || !shouldAcceptRefreshedToken(requestToken, requestGeneration)) {
    return;
  }
  await runWithMediaAuthLock(async () => {
    if (isTokenRefreshThrottled() || !shouldAcceptRefreshedToken(requestToken, requestGeneration)) {
      return;
    }
    try {
      const mediaCookieResponse = await fetch(`${getBaseUrl()}/api/v1/auth/media-cookie`, {
        method: 'POST',
        credentials: 'same-origin',
        headers: { Authorization: `Bearer ${refreshedToken}` },
        // Bound the time this exclusive cross-tab lock is held: login and logout
        // serialize through the same lock, so a black-holed request here must not
        // wedge auth transitions in every tab.
        signal: AbortSignal.timeout(MEDIA_COOKIE_SYNC_TIMEOUT_MS),
      });
      if (mediaCookieResponse.status === 401 || mediaCookieResponse.status === 403) {
        // The server rejected the refreshed token itself — don't commit it.
        return;
      }
    } catch {
      // Cookie sync is best-effort. The token commit below must not depend on it:
      // tokens expire, cookies self-heal (useMediaCookieRefresh, plus the retry on
      // the next refreshed-token header), so dropping the token on a 5xx or network
      // failure would trade a transiently stale media cookie for a hard session
      // expiry mid-activity.
    }
    if (shouldAcceptRefreshedToken(requestToken, requestGeneration)) {
      markTokenRefreshAccepted();
      dispatch(tokenRefreshed(refreshedToken));
    }
  });
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
