import { $queueId } from 'app/store/nanostores/queueId';
import queryString from 'query-string';
import type { components, paths } from 'services/api/schema';

import type { ApiTagDescription } from '..';
import { api, buildV1Url, LIST_ALL_TAG, LIST_TAG } from '..';

/**
 * Builds an endpoint URL for the queue router
 * @example
 * buildQueueUrl('some-path')
 * // '/api/v1/queue/queue_id/some-path'
 */
const buildQueueUrl = (path: string = '') => buildV1Url(`queue/${$queueId.get()}/${path}`);

const getListQueueItemsUrl = (queryArgs?: paths['/api/v1/queue/{queue_id}/list']['get']['parameters']['query']) => {
  const query = queryArgs
    ? queryString.stringify(queryArgs, {
        arrayFormat: 'none',
      })
    : undefined;

  if (query) {
    return buildQueueUrl(`list?${query}`);
  }

  return buildQueueUrl('list');
};

export type SessionQueueItemStatus = NonNullable<
  NonNullable<paths['/api/v1/queue/{queue_id}/list']['get']['parameters']['query']>['status']
>;

export const queueApi = api.injectEndpoints({
  endpoints: (build) => ({
    enqueueBatch: build.mutation<
      paths['/api/v1/queue/{queue_id}/enqueue_batch']['post']['responses']['201']['content']['application/json'],
      paths['/api/v1/queue/{queue_id}/enqueue_batch']['post']['requestBody']['content']['application/json']
    >({
      query: (arg) => ({
        url: buildQueueUrl('enqueue_batch'),
        body: arg,
        method: 'POST',
      }),
      invalidatesTags: [
        'SessionQueueStatus',
        'CurrentSessionQueueItem',
        'NextSessionQueueItem',
        'QueueCountsByDestination',
        { type: 'SessionQueueItem', id: LIST_TAG },
        { type: 'SessionQueueItem', id: LIST_ALL_TAG },
      ],
    }),
    resumeProcessor: build.mutation<
      paths['/api/v1/queue/{queue_id}/processor/resume']['put']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: buildQueueUrl('processor/resume'),
        method: 'PUT',
      }),
      invalidatesTags: ['CurrentSessionQueueItem', 'SessionQueueStatus'],
    }),
    pauseProcessor: build.mutation<
      paths['/api/v1/queue/{queue_id}/processor/pause']['put']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: buildQueueUrl('processor/pause'),
        method: 'PUT',
      }),
      invalidatesTags: ['CurrentSessionQueueItem', 'SessionQueueStatus'],
    }),
    pruneQueue: build.mutation<
      paths['/api/v1/queue/{queue_id}/prune']['put']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: buildQueueUrl('prune'),
        method: 'PUT',
      }),
      invalidatesTags: ['SessionQueueStatus', 'BatchStatus', { type: 'SessionQueueItem', id: LIST_TAG }],
    }),
    clearQueue: build.mutation<
      paths['/api/v1/queue/{queue_id}/clear']['put']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: buildQueueUrl('clear'),
        method: 'PUT',
      }),
      invalidatesTags: [
        'SessionQueueStatus',
        'SessionProcessorStatus',
        'BatchStatus',
        'CurrentSessionQueueItem',
        'NextSessionQueueItem',
        'QueueCountsByDestination',
        { type: 'SessionQueueItem', id: LIST_TAG },
        { type: 'SessionQueueItem', id: LIST_ALL_TAG },
      ],
    }),
    getCurrentQueueItem: build.query<
      paths['/api/v1/queue/{queue_id}/current']['get']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: buildQueueUrl('current'),
        method: 'GET',
      }),
      providesTags: (result) => {
        const tags: ApiTagDescription[] = ['CurrentSessionQueueItem', 'FetchOnReconnect'];
        if (result) {
          tags.push({ type: 'SessionQueueItem', id: result.item_id });
        }
        return tags;
      },
    }),
    getNextQueueItem: build.query<
      paths['/api/v1/queue/{queue_id}/next']['get']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: buildQueueUrl('next'),
        method: 'GET',
      }),
      providesTags: (result) => {
        const tags: ApiTagDescription[] = ['NextSessionQueueItem', 'FetchOnReconnect'];
        if (result) {
          tags.push({ type: 'SessionQueueItem', id: result.item_id });
        }
        return tags;
      },
    }),
    getQueueStatus: build.query<
      paths['/api/v1/queue/{queue_id}/status']['get']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: buildQueueUrl('status'),
        method: 'GET',
      }),
      providesTags: ['SessionQueueStatus', 'FetchOnReconnect'],
    }),
    getBatchStatus: build.query<
      paths['/api/v1/queue/{queue_id}/b/{batch_id}/status']['get']['responses']['200']['content']['application/json'],
      { batch_id: string }
    >({
      query: ({ batch_id }) => ({
        url: buildQueueUrl(`b/${batch_id}/status`),
        method: 'GET',
      }),
      providesTags: (result) => {
        const tags: ApiTagDescription[] = ['FetchOnReconnect'];
        if (result) {
          tags.push({ type: 'BatchStatus', id: result.batch_id });
        }
        return tags;
      },
    }),
    getQueueItem: build.query<
      paths['/api/v1/queue/{queue_id}/i/{item_id}']['get']['responses']['200']['content']['application/json'],
      number
    >({
      query: (item_id) => ({
        url: buildQueueUrl(`i/${item_id}`),
        method: 'GET',
      }),
      providesTags: (result) => {
        const tags: ApiTagDescription[] = ['FetchOnReconnect'];
        if (result) {
          tags.push({ type: 'SessionQueueItem', id: result.item_id });
        }
        return tags;
      },
    }),
    cancelQueueItem: build.mutation<
      paths['/api/v1/queue/{queue_id}/i/{item_id}/cancel']['put']['responses']['200']['content']['application/json'],
      number
    >({
      query: (item_id) => ({
        url: buildQueueUrl(`i/${item_id}/cancel`),
        method: 'PUT',
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        const tags: ApiTagDescription[] = [
          { type: 'SessionQueueItem', id: result.item_id },
          { type: 'BatchStatus', id: result.batch_id },
        ];
        if (result.destination) {
          tags.push({ type: 'QueueCountsByDestination', id: result.destination });
        }
        return tags;
      },
    }),
    cancelByBatchIds: build.mutation<
      paths['/api/v1/queue/{queue_id}/cancel_by_batch_ids']['put']['responses']['200']['content']['application/json'],
      paths['/api/v1/queue/{queue_id}/cancel_by_batch_ids']['put']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildQueueUrl('cancel_by_batch_ids'),
        method: 'PUT',
        body,
      }),
      invalidatesTags: (result, error, { batch_ids }) => {
        if (!result) {
          return [];
        }
        return [
          'SessionQueueStatus',
          'BatchStatus',
          'QueueCountsByDestination',
          { type: 'SessionQueueItem', id: LIST_TAG },
          { type: 'SessionQueueItem', id: LIST_ALL_TAG },
          ...batch_ids.map((id) => ({ type: 'BatchStatus', id }) satisfies ApiTagDescription),
        ];
      },
    }),
    cancelByBatchDestination: build.mutation<
      paths['/api/v1/queue/{queue_id}/cancel_by_destination']['put']['responses']['200']['content']['application/json'],
      paths['/api/v1/queue/{queue_id}/cancel_by_destination']['put']['parameters']['query']
    >({
      query: (params) => ({
        url: buildQueueUrl('cancel_by_destination'),
        method: 'PUT',
        params,
      }),
      invalidatesTags: (result, error, { destination }) => {
        if (!result) {
          return [];
        }
        return [
          'SessionQueueStatus',
          'BatchStatus',
          { type: 'SessionQueueItem', id: LIST_TAG },
          { type: 'SessionQueueItem', id: LIST_ALL_TAG },
          { type: 'QueueCountsByDestination', id: destination },
        ];
      },
    }),
    cancelAllExceptCurrent: build.mutation<
      paths['/api/v1/queue/{queue_id}/cancel_all_except_current']['put']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: buildQueueUrl('cancel_all_except_current'),
        method: 'PUT',
      }),
      invalidatesTags: ['SessionQueueStatus', 'BatchStatus', 'QueueCountsByDestination', 'SessionQueueItem'],
    }),
    retryItemsById: build.mutation<
      paths['/api/v1/queue/{queue_id}/retry_items_by_id']['put']['responses']['200']['content']['application/json'],
      paths['/api/v1/queue/{queue_id}/retry_items_by_id']['put']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildQueueUrl('retry_items_by_id'),
        method: 'PUT',
        body,
      }),
      invalidatesTags: (result, error, item_ids) => {
        if (!result) {
          return [];
        }
        return [
          'CurrentSessionQueueItem',
          'NextSessionQueueItem',
          'QueueCountsByDestination',
          { type: 'SessionQueueItem', id: LIST_TAG },
          { type: 'SessionQueueItem', id: LIST_ALL_TAG },
          ...item_ids.map((id) => ({ type: 'SessionQueueItem', id }) satisfies ApiTagDescription),
        ];
      },
    }),
    listQueueItems: build.query<
      components['schemas']['CursorPaginatedResults_SessionQueueItem_'],
      { cursor?: number; priority?: number; destination?: string } | undefined
    >({
      query: (queryArgs) => ({
        url: getListQueueItemsUrl(queryArgs),
        method: 'GET',
      }),
      keepUnusedDataFor: 60 * 5, // 5 minutes
      providesTags: (result, _error, _args) => {
        if (!result) {
          return [];
        }
        return [
          'FetchOnReconnect',
          { type: 'SessionQueueItem', id: LIST_TAG },
          ...result.items.map(({ item_id }) => ({ type: 'SessionQueueItem', id: item_id }) satisfies ApiTagDescription),
        ];
      },
    }),
    listAllQueueItems: build.query<
      paths['/api/v1/queue/{queue_id}/list_all']['get']['responses']['200']['content']['application/json'],
      paths['/api/v1/queue/{queue_id}/list_all']['get']['parameters']['query']
    >({
      query: (queryArgs) => {
        const q = queryArgs
          ? queryString.stringify(queryArgs, {
              arrayFormat: 'none',
            })
          : undefined;

        return q ? buildQueueUrl(`list_all?${q}`) : buildQueueUrl('list_all');
      },
      providesTags: (result, _error, _args) => {
        if (!result) {
          return [];
        }
        const tags: ApiTagDescription[] = [
          'FetchOnReconnect',
          { type: 'SessionQueueItem', id: LIST_ALL_TAG },
          ...result.map(({ item_id }) => ({ type: 'SessionQueueItem', id: item_id }) satisfies ApiTagDescription),
        ];
        return tags;
      },
    }),
    getQueueCountsByDestination: build.query<
      paths['/api/v1/queue/{queue_id}/counts_by_destination']['get']['responses']['200']['content']['application/json'],
      paths['/api/v1/queue/{queue_id}/counts_by_destination']['get']['parameters']['query']
    >({
      query: (params) => ({ url: buildQueueUrl('counts_by_destination'), method: 'GET', params }),
      providesTags: (result, error, { destination }) => [
        'FetchOnReconnect',
        { type: 'QueueCountsByDestination', id: destination },
      ],
    }),
  }),
});

export const {
  useCancelAllExceptCurrentMutation,
  useCancelByBatchIdsMutation,
  useEnqueueBatchMutation,
  usePauseProcessorMutation,
  useResumeProcessorMutation,
  useClearQueueMutation,
  usePruneQueueMutation,
  useGetQueueStatusQuery,
  useGetQueueItemQuery,
  useListQueueItemsQuery,
  useListAllQueueItemsQuery,
  useCancelQueueItemMutation,
  useGetBatchStatusQuery,
  useGetCurrentQueueItemQuery,
  useGetQueueCountsByDestinationQuery,
  useRetryItemsByIdMutation,
} = queueApi;

export const selectQueueStatus = queueApi.endpoints.getQueueStatus.select();
export const selectCanvasQueueCounts = queueApi.endpoints.getQueueCountsByDestination.select({ destination: 'canvas' });

export const enqueueMutationFixedCacheKeyOptions = {
  fixedCacheKey: 'enqueueBatch',
} as const;
