import queryString from 'query-string';
import type { paths } from 'services/api/schema';

import type { ApiTagDescription } from '..';
import { api, buildV1Url, LIST_TAG } from '..';

/**
 * Builds an endpoint URL for the workflows router
 * @example
 * buildWorkflowsUrl('some-path')
 * // '/api/v1/workflows/some-path'
 */
const buildWorkflowsUrl = (path: string = '') => buildV1Url(`workflows/${path}`);

export const workflowsApi = api.injectEndpoints({
  endpoints: (build) => ({
    getWorkflow: build.query<
      paths['/api/v1/workflows/i/{workflow_id}']['get']['responses']['200']['content']['application/json'],
      string
    >({
      query: (workflow_id) => buildWorkflowsUrl(`i/${workflow_id}`),
      providesTags: (result, error, workflow_id) => [{ type: 'Workflow', id: workflow_id }, 'FetchOnReconnect'],
    }),
    deleteWorkflow: build.mutation<void, string>({
      query: (workflow_id) => ({
        url: buildWorkflowsUrl(`i/${workflow_id}`),
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, workflow_id) => [
        // Because this may change the order of the list, we need to invalidate the whole list
        { type: 'Workflow', id: LIST_TAG },
        { type: 'Workflow', id: workflow_id },
        'WorkflowTagCounts',
        'WorkflowCategoryCounts',
      ],
    }),
    createWorkflow: build.mutation<
      paths['/api/v1/workflows/']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/workflows/']['post']['requestBody']['content']['application/json']['workflow']
    >({
      query: (workflow) => ({
        url: buildWorkflowsUrl(),
        method: 'POST',
        body: { workflow },
      }),
      invalidatesTags: [
        // Because this may change the order of the list, we need to invalidate the whole list
        { type: 'Workflow', id: LIST_TAG },
        'WorkflowTagCounts',
        'WorkflowCategoryCounts',
      ],
    }),
    updateWorkflow: build.mutation<
      paths['/api/v1/workflows/i/{workflow_id}']['patch']['responses']['200']['content']['application/json'],
      paths['/api/v1/workflows/i/{workflow_id}']['patch']['requestBody']['content']['application/json']['workflow']
    >({
      query: (workflow) => ({
        url: buildWorkflowsUrl(`i/${workflow.id}`),
        method: 'PATCH',
        body: { workflow },
      }),
      invalidatesTags: (response, error, workflow) => [
        { type: 'Workflow', id: workflow.id },
        'WorkflowTagCounts',
        'WorkflowCategoryCounts',
      ],
    }),
    getCountsByTag: build.query<
      paths['/api/v1/workflows/counts_by_tag']['get']['responses']['200']['content']['application/json'],
      NonNullable<paths['/api/v1/workflows/counts_by_tag']['get']['parameters']['query']>
    >({
      query: (params) => ({
        url: `${buildWorkflowsUrl('counts_by_tag')}?${queryString.stringify(params, { arrayFormat: 'none' })}`,
      }),
      providesTags: ['WorkflowTagCounts'],
    }),
    getCountsByCategory: build.query<
      paths['/api/v1/workflows/counts_by_category']['get']['responses']['200']['content']['application/json'],
      NonNullable<paths['/api/v1/workflows/counts_by_category']['get']['parameters']['query']>
    >({
      query: (params) => ({
        url: `${buildWorkflowsUrl('counts_by_category')}?${queryString.stringify(params, { arrayFormat: 'none' })}`,
      }),
      providesTags: ['WorkflowCategoryCounts'],
    }),
    listWorkflowsInfinite: build.infiniteQuery<
      paths['/api/v1/workflows/']['get']['responses']['200']['content']['application/json'],
      NonNullable<paths['/api/v1/workflows/']['get']['parameters']['query']>,
      number
    >({
      query: ({ queryArg, pageParam }) => ({
        url: `${buildWorkflowsUrl()}?${queryString.stringify({ ...queryArg, page: pageParam }, { arrayFormat: 'none' })}`,
      }),
      infiniteQueryOptions: {
        initialPageParam: 0,
        getNextPageParam: (_lastPage, _allPages, lastPageParam, _allPageParams) => {
          const finalPage = _lastPage.pages - 1;
          const remainingPages = finalPage - lastPageParam;
          if (remainingPages > 0) {
            return lastPageParam + 1;
          }
          return undefined;
        },
        getPreviousPageParam: (_firstPage, _allPages, firstPageParam, _allPageParams) => {
          return firstPageParam > -1 ? firstPageParam - 1 : undefined;
        },
      },
      providesTags: (result) => {
        const tags: ApiTagDescription[] = ['FetchOnReconnect', { type: 'Workflow', id: LIST_TAG }];
        if (result) {
          tags.push(
            ...result.pages
              .map(({ items }) => items)
              .flat()
              .map((workflow) => ({ type: 'Workflow', id: workflow.workflow_id }) as const)
          );
        }
        return tags;
      },
    }),
    updateOpenedAt: build.mutation<void, { workflow_id: string }>({
      query: ({ workflow_id }) => ({
        url: buildWorkflowsUrl(`i/${workflow_id}/opened_at`),
        method: 'PUT',
      }),
      invalidatesTags: (result, error, { workflow_id }) => [
        { type: 'Workflow', id: workflow_id },
        // Because this may change the order of the list, we need to invalidate the whole list
        { type: 'Workflow', id: LIST_TAG },
      ],
    }),
    setWorkflowThumbnail: build.mutation<void, { workflow_id: string; image: File }>({
      query: ({ workflow_id, image }) => {
        const formData = new FormData();
        formData.append('image', image);
        return {
          url: buildWorkflowsUrl(`i/${workflow_id}/thumbnail`),
          method: 'PUT',
          body: formData,
        };
      },
      invalidatesTags: (result, error, { workflow_id }) => [{ type: 'Workflow', id: workflow_id }],
    }),
    deleteWorkflowThumbnail: build.mutation<void, string>({
      query: (workflow_id) => ({
        url: buildWorkflowsUrl(`i/${workflow_id}/thumbnail`),
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, workflow_id) => [{ type: 'Workflow', id: workflow_id }],
    }),
    unpublishWorkflow: build.mutation<void, string>({
      query: (workflow_id) => ({
        url: buildWorkflowsUrl(`i/${workflow_id}/unpublish`),
        method: 'POST',
      }),
      invalidatesTags: (result, error, workflow_id) => [{ type: 'Workflow', id: workflow_id }],
    }),
  }),
});

export const {
  useUpdateOpenedAtMutation,
  useGetCountsByTagQuery,
  useGetCountsByCategoryQuery,
  useLazyGetWorkflowQuery,
  useGetWorkflowQuery,
  useCreateWorkflowMutation,
  useDeleteWorkflowMutation,
  useUpdateWorkflowMutation,
  useListWorkflowsInfiniteInfiniteQuery,
  useSetWorkflowThumbnailMutation,
  useDeleteWorkflowThumbnailMutation,
  useUnpublishWorkflowMutation,
} = workflowsApi;
