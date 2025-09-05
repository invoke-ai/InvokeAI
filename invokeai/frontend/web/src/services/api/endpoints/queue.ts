import { $queueId } from 'app/store/nanostores/queueId';
import queryString from 'query-string';
import type { components, paths } from 'services/api/schema';
import type {
  GetQueueItemDTOsByItemIdsArgs,
  GetQueueItemDTOsByItemIdsResult,
  GetQueueItemIdsArgs,
  GetQueueItemIdsResult,
} from 'services/api/types';
import stableHash from 'stable-hash';
import type { Param0 } from 'tsafe';

import type { ApiTagDescription } from '..';
import { api, buildV1Url, LIST_ALL_TAG, LIST_TAG } from '..';

/**
 * Builds an endpoint URL for the queue router
 * @example
 * buildQueueUrl('some-path')
 * // '/api/v1/queue/queue_id/some-path'
 */
const buildQueueUrl = (path: string = '') => buildV1Url(`queue/${$queueId.get()}/${path}`);

export type SessionQueueItemStatus = NonNullable<components['schemas']['SessionQueueItem']['status']>;

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
        'CurrentSessionQueueItem',
        'NextSessionQueueItem',
        'QueueCountsByDestination',
        'SessionQueueItemIdList',
        { type: 'SessionQueueItem', id: LIST_TAG },
        { type: 'SessionQueueItem', id: LIST_ALL_TAG },
      ],
      onQueryStarted: async (arg, api) => {
        const { dispatch, queryFulfilled } = api;
        try {
          const { data } = await queryFulfilled;
          dispatch(
            queueApi.util.updateQueryData('getQueueStatus', undefined, (draft) => {
              draft.queue.in_progress += data.item_ids.length;
              draft.queue.total += data.item_ids.length;
            })
          );
        } catch {
          // no-op
        }
      },
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
      invalidatesTags: [
        'SessionQueueStatus',
        'BatchStatus',
        'SessionQueueItemIdList',
        { type: 'SessionQueueItem', id: LIST_TAG },
        { type: 'SessionQueueItem', id: LIST_ALL_TAG },
      ],
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
        'SessionQueueItemIdList',
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
          tags.push({ type: 'BatchStatus', id: result.batch_id });
        }
        return tags;
      },
    }),
    cancelQueueItem: build.mutation<
      paths['/api/v1/queue/{queue_id}/i/{item_id}/cancel']['put']['responses']['200']['content']['application/json'],
      { item_id: number }
    >({
      query: ({ item_id }) => ({
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
    cancelQueueItemsByDestination: build.mutation<
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
    getQueueItemIds: build.query<GetQueueItemIdsResult, GetQueueItemIdsArgs>({
      query: (queryArgs) => ({
        url: buildQueueUrl(`item_ids?${queryString.stringify(queryArgs)}`),
        method: 'GET',
      }),
      providesTags: (queryArgs) => [
        'FetchOnReconnect',
        'SessionQueueItemIdList',
        { type: 'SessionQueueItemIdList', id: stableHash(queryArgs) },
      ],
    }),
    getQueueItemDTOsByItemIds: build.mutation<GetQueueItemDTOsByItemIdsResult, GetQueueItemDTOsByItemIdsArgs>({
      query: (body) => ({
        url: buildQueueUrl('items_by_ids'),
        method: 'POST',
        body,
      }),
      // Don't provide cache tags - we'll manually upsert into individual getQueueItem caches
      async onQueryStarted(_, { dispatch, queryFulfilled }) {
        try {
          const { data: queueItems } = await queryFulfilled;

          // Upsert each queue item into the individual item cache
          const updates: Param0<typeof queueApi.util.upsertQueryEntries> = [];
          for (const queueItem of queueItems) {
            updates.push({
              endpointName: 'getQueueItem',
              arg: queueItem.item_id,
              value: queueItem,
            });
          }
          dispatch(queueApi.util.upsertQueryEntries(updates));
        } catch {
          // Handle error if needed
        }
      },
    }),
    deleteQueueItem: build.mutation<void, { item_id: number }>({
      query: ({ item_id }) => ({
        url: buildQueueUrl(`i/${item_id}`),
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, { item_id }) => [
        { type: 'SessionQueueItem', id: item_id },
        { type: 'SessionQueueItem', id: LIST_TAG },
        { type: 'SessionQueueItem', id: LIST_ALL_TAG },
      ],
    }),
    deleteQueueItemsByDestination: build.mutation<void, { destination: string }>({
      query: ({ destination }) => ({
        url: buildQueueUrl(`d/${destination}`),
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, { destination }) => [
        { type: 'QueueCountsByDestination', id: destination },
        { type: 'SessionQueueItem', id: LIST_TAG },
        { type: 'SessionQueueItem', id: LIST_ALL_TAG },
      ],
    }),
    deleteAllExceptCurrent: build.mutation<
      paths['/api/v1/queue/{queue_id}/delete_all_except_current']['put']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: buildQueueUrl('delete_all_except_current'),
        method: 'PUT',
      }),
      invalidatesTags: ['SessionQueueStatus', 'BatchStatus', 'QueueCountsByDestination', 'SessionQueueItem'],
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
  useGetQueueItemIdsQuery,
  useGetQueueItemDTOsByItemIdsMutation,
  useCancelQueueItemMutation,
  useCancelQueueItemsByDestinationMutation,
  useGetCurrentQueueItemQuery,

  useDeleteQueueItemMutation,
  useDeleteAllExceptCurrentMutation,
  useGetBatchStatusQuery,

  useGetQueueCountsByDestinationQuery,
  useRetryItemsByIdMutation,
} = queueApi;

export const selectQueueStatus = queueApi.endpoints.getQueueStatus.select();

export const enqueueMutationFixedCacheKeyOptions = {
  fixedCacheKey: 'enqueueBatch',
} as const;

export const useIsGenerationInProgress = () => {
  const { data } = useGetQueueStatusQuery();
  return data && data.queue.in_progress > 0;
};
