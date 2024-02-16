import type { paths } from 'services/api/schema';

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
      onQueryStarted: async (arg, api) => {
        const { dispatch, queryFulfilled } = api;
        try {
          await queryFulfilled;
          dispatch(workflowsApi.util.invalidateTags([{ type: 'WorkflowsRecent', id: LIST_TAG }]));
        } catch {
          // no-op
        }
      },
    }),
    deleteWorkflow: build.mutation<void, string>({
      query: (workflow_id) => ({
        url: buildWorkflowsUrl(`i/${workflow_id}`),
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, workflow_id) => [
        { type: 'Workflow', id: LIST_TAG },
        { type: 'Workflow', id: workflow_id },
        { type: 'WorkflowsRecent', id: LIST_TAG },
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
        { type: 'Workflow', id: LIST_TAG },
        { type: 'WorkflowsRecent', id: LIST_TAG },
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
        { type: 'WorkflowsRecent', id: LIST_TAG },
        { type: 'Workflow', id: LIST_TAG },
        { type: 'Workflow', id: workflow.id },
      ],
    }),
    listWorkflows: build.query<
      paths['/api/v1/workflows/']['get']['responses']['200']['content']['application/json'],
      NonNullable<paths['/api/v1/workflows/']['get']['parameters']['query']>
    >({
      query: (params) => ({
        url: buildWorkflowsUrl(),
        params,
      }),
      providesTags: ['FetchOnReconnect', { type: 'Workflow', id: LIST_TAG }],
    }),
  }),
});

export const {
  useLazyGetWorkflowQuery,
  useCreateWorkflowMutation,
  useDeleteWorkflowMutation,
  useUpdateWorkflowMutation,
  useListWorkflowsQuery,
} = workflowsApi;
