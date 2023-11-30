import { WorkflowV2 } from 'features/nodes/types/workflow';
import { paths } from 'services/api/schema';
import { LIST_TAG, api } from '..';

export const workflowsApi = api.injectEndpoints({
  endpoints: (build) => ({
    getWorkflow: build.query<
      paths['/api/v1/workflows/i/{workflow_id}']['get']['responses']['200']['content']['application/json'],
      string
    >({
      query: (workflow_id) => `workflows/i/${workflow_id}`,
      providesTags: (result, error, workflow_id) => [
        { type: 'Workflow', id: workflow_id },
      ],
    }),
    deleteWorkflow: build.mutation<void, string>({
      query: (workflow_id) => ({
        url: `workflows/i/${workflow_id}`,
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, workflow_id) => [
        { type: 'Workflow', id: LIST_TAG },
        { type: 'Workflow', id: workflow_id },
      ],
    }),
    createWorkflow: build.mutation<
      paths['/api/v1/workflows/']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/workflows/']['post']['requestBody']['content']['application/json']['workflow']
    >({
      query: (workflow) => ({
        url: 'workflows',
        method: 'POST',
        body: workflow,
      }),
      invalidatesTags: [{ type: 'Workflow', id: LIST_TAG }],
    }),
    updateWorkflow: build.mutation<
      paths['/api/v1/workflows/i/{workflow_id}']['patch']['responses']['200']['content']['application/json'],
      paths['/api/v1/workflows/i/{workflow_id}']['patch']['requestBody']['content']['application/json']['workflow']
    >({
      query: (workflow) => ({
        url: `workflows/i/${workflow.id}`,
        method: 'PATCH',
        body: workflow,
      }),
      invalidatesTags: (response, error, workflow) => [
        { type: 'Workflow', id: LIST_TAG },
        { type: 'Workflow', id: workflow.id },
      ],
    }),
    listWorkflows: build.query<
      paths['/api/v1/workflows/']['get']['responses']['200']['content']['application/json'],
      NonNullable<paths['/api/v1/workflows/']['get']['parameters']['query']>
    >({
      query: (params) => ({
        url: 'workflows/',
        params,
      }),
      providesTags: [{ type: 'Workflow', id: LIST_TAG }],
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
