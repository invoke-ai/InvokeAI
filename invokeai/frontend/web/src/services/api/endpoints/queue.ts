import { EntityState, createEntityAdapter } from '@reduxjs/toolkit';
import queryString from 'query-string';
import { ApiTagDescription, LIST_TAG, api } from '..';
import { components, paths } from '../schema';

const getListQueueItemsUrl = (
  queryArgs?: paths['/api/v1/queue/list']['get']['parameters']['query']
) =>
  queryArgs
    ? `queue/list?${queryString.stringify(queryArgs, { arrayFormat: 'none' })}`
    : 'queue/list';

export type SessionQueueItemStatus = NonNullable<
  NonNullable<
    paths['/api/v1/queue/list']['get']['parameters']['query']
  >['status']
>;

export const queueItemsAdapter = createEntityAdapter<
  components['schemas']['SessionQueueItemDTO']
>({
  selectId: (queueItem) => queueItem.id,
  sortComparer: (a, b) => {
    // Sort by priority in descending order
    if (a.priority > b.priority) {
      return -1;
    }
    if (a.priority < b.priority) {
      return 1;
    }

    // If priority is the same, sort by id in ascending order
    if (a.id < b.id) {
      return -1;
    }
    if (a.id > b.id) {
      return 1;
    }

    return 0;
  },
});

export const queueApi = api.injectEndpoints({
  endpoints: (build) => ({
    enqueueBatch: build.mutation<
      paths['/api/v1/queue/enqueue_batch']['post']['responses']['201']['content']['application/json'],
      paths['/api/v1/queue/enqueue_batch']['post']['requestBody']['content']['application/json']
    >({
      query: (arg) => ({
        url: 'queue/enqueue_batch',
        body: arg,
        method: 'POST',
      }),
      invalidatesTags: [
        'SessionQueueStatus',
        'CurrentSessionQueueItem',
        'NextSessionQueueItem',
        { type: 'SessionQueueItemDTO', id: LIST_TAG },
      ],
    }),
    enqueueGraph: build.mutation<
      paths['/api/v1/queue/enqueue']['post']['responses']['201']['content']['application/json'],
      paths['/api/v1/queue/enqueue']['post']['requestBody']['content']['application/json']
    >({
      query: (arg) => ({
        url: 'queue/enqueue',
        body: arg,
        method: 'POST',
      }),
      invalidatesTags: [
        'SessionQueueStatus',
        'CurrentSessionQueueItem',
        'NextSessionQueueItem',
        { type: 'SessionQueueItemDTO', id: LIST_TAG },
      ],
    }),
    startQueueExecution: build.mutation<void, void>({
      query: () => ({
        url: 'queue/start',
        method: 'PUT',
      }),
      invalidatesTags: ['SessionQueueStatus'],
    }),
    stopQueueExecution: build.mutation<void, void>({
      query: () => ({
        url: 'queue/stop',
        method: 'PUT',
      }),
      invalidatesTags: ['SessionQueueStatus'],
    }),
    cancelQueueExecution: build.mutation<void, void>({
      query: () => ({
        url: 'queue/cancel',
        method: 'PUT',
      }),
      invalidatesTags: ['SessionQueueStatus'],
    }),
    pruneQueue: build.mutation<
      paths['/api/v1/queue/prune']['put']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: 'queue/prune',
        method: 'PUT',
      }),
      invalidatesTags: [
        'SessionQueueStatus',
        { type: 'SessionQueueItemDTO', id: LIST_TAG },
      ],
      onQueryStarted: async (arg, api) => {
        const { dispatch, queryFulfilled } = api;
        const patch = dispatch(
          queueApi.util.updateQueryData(
            'listQueueItems',
            undefined,
            (draft) => {
              const ids = queueItemsAdapter
                .getSelectors()
                .selectAll(draft)
                .filter((item) =>
                  ['completed', 'failed', 'canceled'].includes(item.status)
                )
                .map((item) => item.id);
              queueItemsAdapter.removeMany(draft, ids);
            }
          )
        );
        try {
          await queryFulfilled;
        } catch {
          patch.undo();
        }
      },
    }),
    clearQueue: build.mutation<
      paths['/api/v1/queue/clear']['put']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: 'queue/clear',
        method: 'PUT',
      }),
      invalidatesTags: [
        'SessionQueueStatus',
        'CurrentSessionQueueItem',
        'NextSessionQueueItem',
        { type: 'SessionQueueItemDTO', id: LIST_TAG },
      ],
      onQueryStarted: async (arg, api) => {
        const { dispatch, queryFulfilled } = api;
        const listPatch = dispatch(
          queueApi.util.updateQueryData(
            'listQueueItems',
            undefined,
            (draft) => {
              queueItemsAdapter.removeAll(draft);
            }
          )
        );
        const statusPatch = dispatch(
          queueApi.util.updateQueryData(
            'getQueueStatus',
            undefined,
            (draft) => {
              draft.started = false;
              draft.canceled = 0;
              draft.completed = 0;
              draft.failed = 0;
              draft.in_progress = 0;
              draft.pending = 0;
              draft.total = 0;
            }
          )
        );
        try {
          await queryFulfilled;
        } catch {
          listPatch.undo();
          statusPatch.undo();
        }
      },
    }),
    getCurrentQueueItem: build.query<
      paths['/api/v1/queue/current']['get']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: 'queue/current',
        method: 'GET',
      }),
      providesTags: (result) => {
        const tags: ApiTagDescription[] = ['CurrentSessionQueueItem'];
        if (result) {
          tags.push({ type: 'SessionQueueItem', id: result.id });
        }
        return tags;
      },
    }),
    peekNextQueueItem: build.query<
      paths['/api/v1/queue/peek']['get']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: 'queue/peek',
        method: 'GET',
      }),
      providesTags: (result) => {
        const tags: ApiTagDescription[] = ['NextSessionQueueItem'];
        if (result) {
          tags.push({ type: 'SessionQueueItem', id: result.id });
        }
        return tags;
      },
    }),
    getQueueStatus: build.query<
      paths['/api/v1/queue/status']['get']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: 'queue/status',
        method: 'GET',
      }),
      providesTags: ['SessionQueueStatus'],
    }),
    getQueueItem: build.query<
      paths['/api/v1/queue/q/{id}']['get']['responses']['200']['content']['application/json'],
      paths['/api/v1/queue/q/{id}']['get']['parameters']['path']['id']
    >({
      query: (id) => ({
        url: `queue/q/${id}`,
        method: 'GET',
      }),
      providesTags: (result) => {
        if (!result) {
          return [];
        }
        return [{ type: 'SessionQueueItem', id: result.id }];
      },
    }),
    cancelQueueItem: build.mutation<
      paths['/api/v1/queue/q/{id}/cancel']['put']['responses']['200']['content']['application/json'],
      paths['/api/v1/queue/q/{id}/cancel']['put']['parameters']['path']['id']
    >({
      query: (id) => ({
        url: `queue/q/${id}/cancel`,
        method: 'GET',
      }),
    }),
    listQueueItems: build.query<
      EntityState<components['schemas']['SessionQueueItemDTO']>,
      { cursor?: number; priority?: number } | undefined
    >({
      query: (queryArgs) => ({
        url: getListQueueItemsUrl(queryArgs),
        method: 'GET',
      }),
      providesTags: (result) => {
        if (!result) {
          return [];
        }
        return [{ type: 'SessionQueueItemDTO', id: LIST_TAG }];
      },
      serializeQueryArgs: () => {
        return 'queue/list';
      },
      transformResponse: (
        response: components['schemas']['CursorPaginatedResults_SessionQueueItemDTO_']
      ) =>
        queueItemsAdapter.addMany(
          queueItemsAdapter.getInitialState(),
          response.items
        ),
      merge: (cache, response) => {
        queueItemsAdapter.addMany(
          cache,
          queueItemsAdapter.getSelectors().selectAll(response)
        );
      },
      forceRefetch: ({ currentArg, previousArg }) => {
        return (
          currentArg?.cursor !== previousArg?.cursor ||
          currentArg?.priority !== previousArg?.priority
        );
      },
    }),
  }),
});

export const {
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
