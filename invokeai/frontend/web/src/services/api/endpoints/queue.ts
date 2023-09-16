import {
  AnyAction,
  EntityState,
  ThunkDispatch,
  createEntityAdapter,
} from '@reduxjs/toolkit';
import queryString from 'query-string';
import { ApiTagDescription, api } from '..';
import { components, paths } from '../schema';
import { $queueId } from 'features/queue/store/nanoStores';
import { listParamsReset } from 'features/queue/store/queueSlice';

const getListQueueItemsUrl = (
  queryArgs?: paths['/api/v1/queue/{queue_id}/list']['get']['parameters']['query']
) => {
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
  NonNullable<
    paths['/api/v1/queue/{queue_id}/list']['get']['parameters']['query']
  >['status']
>;

export const queueItemsAdapter = createEntityAdapter<
  components['schemas']['SessionQueueItemDTO']
>({
  selectId: (queueItem) => queueItem.item_id,
  sortComparer: (a, b) => {
    // Sort by priority in descending order
    if (a.priority > b.priority) {
      return -1;
    }
    if (a.priority < b.priority) {
      return 1;
    }

    // If priority is the same, sort by id in ascending order
    if (a.order_id < b.order_id) {
      return -1;
    }
    if (a.order_id > b.order_id) {
      return 1;
    }

    return 0;
  },
});

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
      invalidatesTags: [
        'SessionQueueStatus',
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
    enqueueGraph: build.mutation<
      paths['/api/v1/queue/{queue_id}/enqueue_graph']['post']['responses']['201']['content']['application/json'],
      paths['/api/v1/queue/{queue_id}/enqueue_graph']['post']['requestBody']['content']['application/json']
    >({
      query: (arg) => ({
        url: `queue/${$queueId.get()}/enqueue_graph`,
        body: arg,
        method: 'POST',
      }),
      invalidatesTags: [
        'SessionQueueStatus',
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
    startQueueExecution: build.mutation<void, void>({
      query: () => ({
        url: `queue/${$queueId.get()}/start`,
        method: 'PUT',
      }),
      invalidatesTags: ['SessionQueueStatus'],
    }),
    stopQueueExecution: build.mutation<void, void>({
      query: () => ({
        url: `queue/${$queueId.get()}/stop`,
        method: 'PUT',
      }),
      invalidatesTags: ['SessionQueueStatus'],
    }),
    cancelQueueExecution: build.mutation<void, void>({
      query: () => ({
        url: `queue/${$queueId.get()}/cancel`,
        method: 'PUT',
      }),
      invalidatesTags: ['SessionQueueStatus'],
    }),
    pruneQueue: build.mutation<
      paths['/api/v1/queue/{queue_id}/prune']['put']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: `queue/${$queueId.get()}/prune`,
        method: 'PUT',
      }),
      invalidatesTags: ['SessionQueueStatus'],
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
        const tags: ApiTagDescription[] = ['CurrentSessionQueueItem'];
        if (result) {
          tags.push({ type: 'SessionQueueItem', id: result.item_id });
        }
        return tags;
      },
    }),
    peekNextQueueItem: build.query<
      paths['/api/v1/queue/{queue_id}/peek']['get']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: `queue/${$queueId.get()}/peek`,
        method: 'GET',
      }),
      providesTags: (result) => {
        const tags: ApiTagDescription[] = ['NextSessionQueueItem'];
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
      providesTags: ['SessionQueueStatus'],
    }),
    getQueueItem: build.query<
      paths['/api/v1/queue/{queue_id}/i/{item_id}']['get']['responses']['200']['content']['application/json'],
      string
    >({
      query: (item_id) => ({
        url: `queue/${$queueId.get()}/i/${item_id}`,
        method: 'GET',
      }),
      providesTags: (result) => {
        if (!result) {
          return [];
        }
        return [{ type: 'SessionQueueItem', id: result.item_id }];
      },
    }),
    cancelQueueItem: build.mutation<
      paths['/api/v1/queue/{queue_id}/i/{item_id}/cancel']['put']['responses']['200']['content']['application/json'],
      string
    >({
      query: (item_id) => ({
        url: `queue/${$queueId.get()}/i/${item_id}/cancel`,
        method: 'GET',
      }),
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
    }),
    listQueueItems: build.query<
      EntityState<components['schemas']['SessionQueueItemDTO']> & {
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
      transformResponse: (
        response: components['schemas']['CursorPaginatedResults_SessionQueueItemDTO_']
      ) =>
        queueItemsAdapter.addMany(
          queueItemsAdapter.getInitialState({
            has_more: response.has_more,
          }),
          response.items
        ),
      merge: (cache, response) => {
        console.log(cache);
        queueItemsAdapter.addMany(
          cache,
          queueItemsAdapter.getSelectors().selectAll(response)
        );
        cache.has_more = response.has_more;
      },
      forceRefetch: ({ currentArg, previousArg }) => currentArg !== previousArg,
      keepUnusedDataFor: 60 * 5, // 5 minutes
    }),
  }),
});

export const {
  useCancelByBatchIdsMutation,
  useEnqueueGraphMutation,
  useEnqueueBatchMutation,
  useCancelQueueExecutionMutation,
  useStopQueueExecutionMutation,
  useStartQueueExecutionMutation,
  useClearQueueMutation,
  usePruneQueueMutation,
  useGetCurrentQueueItemQuery,
  useGetQueueStatusQuery,
  useGetQueueItemQuery,
  usePeekNextQueueItemQuery,
  useListQueueItemsQuery,
} = queueApi;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const resetListQueryData = (dispatch: ThunkDispatch<any, any, AnyAction>) => {
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
