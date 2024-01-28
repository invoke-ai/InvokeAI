import type { EntityState, ThunkDispatch, UnknownAction } from '@reduxjs/toolkit';
import { createEntityAdapter } from '@reduxjs/toolkit';
import { getSelectorsOptions } from 'app/store/createMemoizedSelector';
import { $queueId } from 'app/store/nanostores/queueId';
import { listParamsReset } from 'features/queue/store/queueSlice';
import queryString from 'query-string';
import type { components, paths } from 'services/api/schema';

import type { ApiTagDescription } from '..';
import { api } from '..';

const getListQueueItemsUrl = (queryArgs?: paths['/api/v1/queue/{queue_id}/list']['get']['parameters']['query']) => {
  const query = queryArgs
    ? queryString.stringify(queryArgs, {
        arrayFormat: 'none',
      })
    : undefined;

  if (query) {
    return `queue/${$queueId.get()}/list?${query}`;
  }

  return `queue/${$queueId.get()}/list`;
};

export type SessionQueueItemStatus = NonNullable<
  NonNullable<paths['/api/v1/queue/{queue_id}/list']['get']['parameters']['query']>['status']
>;

export const queueItemsAdapter = createEntityAdapter<components['schemas']['SessionQueueItemDTO'], string>({
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
        url: `queue/${$queueId.get()}/enqueue_batch`,
        body: arg,
        method: 'POST',
      }),
      invalidatesTags: ['SessionQueueStatus', 'CurrentSessionQueueItem', 'NextSessionQueueItem'],
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
    resumeProcessor: build.mutation<
      paths['/api/v1/queue/{queue_id}/processor/resume']['put']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: `queue/${$queueId.get()}/processor/resume`,
        method: 'PUT',
      }),
      invalidatesTags: ['CurrentSessionQueueItem', 'SessionQueueStatus'],
    }),
    pauseProcessor: build.mutation<
      paths['/api/v1/queue/{queue_id}/processor/pause']['put']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: `queue/${$queueId.get()}/processor/pause`,
        method: 'PUT',
      }),
      invalidatesTags: ['CurrentSessionQueueItem', 'SessionQueueStatus'],
    }),
    pruneQueue: build.mutation<
      paths['/api/v1/queue/{queue_id}/prune']['put']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: `queue/${$queueId.get()}/prune`,
        method: 'PUT',
      }),
      invalidatesTags: ['SessionQueueStatus', 'BatchStatus'],
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
        url: `queue/${$queueId.get()}/clear`,
        method: 'PUT',
      }),
      invalidatesTags: [
        'SessionQueueStatus',
        'SessionProcessorStatus',
        'BatchStatus',
        'CurrentSessionQueueItem',
        'NextSessionQueueItem',
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
        url: `queue/${$queueId.get()}/current`,
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
        url: `queue/${$queueId.get()}/next`,
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
        url: `queue/${$queueId.get()}/status`,
        method: 'GET',
      }),
      providesTags: ['SessionQueueStatus', 'FetchOnReconnect'],
    }),
    getBatchStatus: build.query<
      paths['/api/v1/queue/{queue_id}/b/{batch_id}/status']['get']['responses']['200']['content']['application/json'],
      { batch_id: string }
    >({
      query: ({ batch_id }) => ({
        url: `queue/${$queueId.get()}/b/${batch_id}/status`,
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
        url: `queue/${$queueId.get()}/i/${item_id}`,
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
        url: `queue/${$queueId.get()}/i/${item_id}/cancel`,
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
        return [
          { type: 'SessionQueueItem', id: result.item_id },
          { type: 'BatchStatus', id: result.batch_id },
        ];
      },
    }),
    cancelByBatchIds: build.mutation<
      paths['/api/v1/queue/{queue_id}/cancel_by_batch_ids']['put']['responses']['200']['content']['application/json'],
      paths['/api/v1/queue/{queue_id}/cancel_by_batch_ids']['put']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: `queue/${$queueId.get()}/cancel_by_batch_ids`,
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
      invalidatesTags: ['SessionQueueStatus', 'BatchStatus'],
    }),
    listQueueItems: build.query<
      EntityState<components['schemas']['SessionQueueItemDTO'], string> & {
        has_more: boolean;
      },
      { cursor?: number; priority?: number } | undefined
    >({
      query: (queryArgs) => ({
        url: getListQueueItemsUrl(queryArgs),
        method: 'GET',
      }),
      serializeQueryArgs: () => {
        return `queue/${$queueId.get()}/list`;
      },
      transformResponse: (response: components['schemas']['CursorPaginatedResults_SessionQueueItemDTO_']) =>
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
      providesTags: ['FetchOnReconnect'],
    }),
  }),
});

export const {
  useCancelByBatchIdsMutation,
  useEnqueueBatchMutation,
  usePauseProcessorMutation,
  useResumeProcessorMutation,
  useClearQueueMutation,
  usePruneQueueMutation,
  useGetCurrentQueueItemQuery,
  useGetQueueStatusQuery,
  useGetQueueItemQuery,
  useGetNextQueueItemQuery,
  useListQueueItemsQuery,
  useCancelQueueItemMutation,
  useGetBatchStatusQuery,
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
