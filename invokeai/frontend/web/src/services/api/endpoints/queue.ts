import type { EntityState, ThunkDispatch, UnknownAction } from '@reduxjs/toolkit';
import { createEntityAdapter } from '@reduxjs/toolkit';
import { getSelectorsOptions } from 'app/store/createMemoizedSelector';
import { $queueId } from 'app/store/nanostores/queueId';
import { listParamsReset } from 'features/queue/store/queueSlice';
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

export const queueItemsAdapter = createEntityAdapter<components['schemas']['SessionQueueItem'], string>({
  selectId: (queueItem) => String(queueItem.item_id),
  sortComparer: (a, b) => {
    // Sort by priority in descending order
    if (a.priority > b.priority) {
      return -1;
    }
    if (a.priority < b.priority) {
      return 1;
    }

    // If priority is the same, sort by id in ascending order
    if (a.item_id < b.item_id) {
      return -1;
    }
    if (a.item_id > b.item_id) {
      return 1;
    }

    return 0;
  },
});
export const queueItemsAdapterSelectors = queueItemsAdapter.getSelectors(undefined, getSelectorsOptions);

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
      onQueryStarted: async (arg, api) => {
        const { dispatch, queryFulfilled } = api;
        try {
          const { data } = await queryFulfilled;
          resetListQueryData(dispatch);
          /**
           * When a batch is enqueued, we need to update the queue status. While it might be templting to invalidate the
           * `SessionQueueStatus` tag here, this can introduce a race condition when the queue item executes quickly:
           *
           * - Enqueue via this query
           * - On success, we invalidate `SessionQueueStatus` tag - network request sent to server
           * - The server gets the queue status request and responds, but this takes some time... in the meantime:
           *   - The new queue item starts executing, and we receive a socket queue item status changed event
           *   - We optimistically update the queue status in the queue item status changed socket handler
           *   - At this point, the queue status is correct
           * - Finally, we get the queue status from the tag invalidation request - but it's reporting the queue status
           *   from _before_ the last queue event
           * - The queue status is now incorrect!
           *
           * Ok, what if we just never did optimistic updates and invalidated the tag in the queue event handlers instead?
           * It's much simpler that way, but it causes a lot of network requests - 3 per queue item, as it moves from
           * pending -> in_progress -> completed/failed/canceled.
           *
           * We can do a bit of extra work here, incrementing the pending and total counts in the queue status, and do
           * similar optimistic updates in the socket handler. Because this optimistic update runs immediately after the
           * enqueue network request, it should always occur _before_ the next queue event, so no race condition:
           *
           * - Enqueue batch via this query
           * - On success, optimistically update - this happens immediately on the HTTP OK - before the next queue event
           * - At this point, the queue status is correct
           * - A queue item status changes and we receive a socket event w/ updated status
           * - Update status optimistically in socket handler
           * - Queue status is still correct
           *
           * This problem occurs most commonly with canvas filters like Canny edge detection, which are single-node
           * graphs that execute very quickly. Image generation graphs take long enough to not trigger this race
           * condition - even when all nodes are cached on the server.
           */
          dispatch(
            queueApi.util.updateQueryData('getQueueStatus', undefined, (draft) => {
              if (!draft) {
                return;
              }
              draft.queue.pending += data.enqueued;
              draft.queue.total += data.enqueued;
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
        { type: 'SessionQueueItem', id: LIST_TAG },
        { type: 'SessionQueueItem', id: LIST_ALL_TAG },
      ],
      onQueryStarted: async (arg, api) => {
        const { dispatch, queryFulfilled } = api;
        try {
          await queryFulfilled;
          resetListQueryData(dispatch);
        } catch {
          // no-op
        }
      },
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
      onQueryStarted: async (arg, api) => {
        const { dispatch, queryFulfilled } = api;
        try {
          await queryFulfilled;
          resetListQueryData(dispatch);
        } catch {
          // no-op
        }
      },
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
      { item_id: number }
    >({
      query: ({ item_id }) => ({
        url: buildQueueUrl(`i/${item_id}/cancel`),
        method: 'PUT',
      }),
      onQueryStarted: async (item_id, { dispatch, queryFulfilled }) => {
        try {
          const { data } = await queryFulfilled;
          dispatch(
            queueApi.util.updateQueryData('listQueueItems', undefined, (draft) => {
              queueItemsAdapter.updateOne(draft, {
                id: String(item_id),
                changes: {
                  status: data.status,
                  completed_at: data.completed_at,
                  updated_at: data.updated_at,
                },
              });
            })
          );
        } catch {
          // no-op
        }
      },
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
      onQueryStarted: async (arg, api) => {
        const { dispatch, queryFulfilled } = api;
        try {
          await queryFulfilled;
          resetListQueryData(dispatch);
        } catch {
          // no-op
        }
      },
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
      onQueryStarted: async (arg, api) => {
        const { dispatch, queryFulfilled } = api;
        try {
          await queryFulfilled;
          resetListQueryData(dispatch);
        } catch {
          // no-op
        }
      },
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
      EntityState<components['schemas']['SessionQueueItem'], string> & {
        has_more: boolean;
      },
      { cursor?: number; priority?: number } | undefined
    >({
      query: (queryArgs) => ({
        url: getListQueueItemsUrl(queryArgs),
        method: 'GET',
      }),
      serializeQueryArgs: () => {
        return buildQueueUrl('list');
      },
      transformResponse: (response: components['schemas']['CursorPaginatedResults_SessionQueueItem_']) =>
        queueItemsAdapter.addMany(
          queueItemsAdapter.getInitialState({
            has_more: response.has_more,
          }),
          response.items
        ),
      merge: (cache, response) => {
        queueItemsAdapter.addMany(cache, queueItemsAdapterSelectors.selectAll(response));
        cache.has_more = response.has_more;
      },
      forceRefetch: ({ currentArg, previousArg }) => currentArg !== previousArg,
      keepUnusedDataFor: 60 * 5, // 5 minutes
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
  useListQueueItemsQuery,
  useCancelQueueItemMutation,
  useCancelQueueItemsByDestinationMutation,

  useDeleteQueueItemMutation,
  useDeleteAllExceptCurrentMutation,
  useGetBatchStatusQuery,

  useGetQueueCountsByDestinationQuery,
  useRetryItemsByIdMutation,
} = queueApi;

export const selectQueueStatus = queueApi.endpoints.getQueueStatus.select();

const resetListQueryData = (
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  dispatch: ThunkDispatch<any, any, UnknownAction>
) => {
  dispatch(
    queueApi.util.updateQueryData('listQueueItems', undefined, (draft) => {
      // remove all items from the list
      queueItemsAdapter.removeAll(draft);
      // reset the has_more flag
      draft.has_more = false;
    })
  );
  // set the list cursor and priority to undefined
  dispatch(listParamsReset());
  // we have to manually kick off another query to get the first page and re-initialize the list
  dispatch(queueApi.endpoints.listQueueItems.initiate(undefined));
};

export const enqueueMutationFixedCacheKeyOptions = {
  fixedCacheKey: 'enqueueBatch',
} as const;

export const useIsGenerationInProgress = () => {
  const { data } = useGetQueueStatusQuery();
  return data && data.queue.in_progress > 0;
};
