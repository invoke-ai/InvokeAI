import { logger } from 'app/logging/logger';
import { WorkflowV2, zWorkflowV2 } from 'features/nodes/types/workflow';
import { api } from '..';
import { paths } from 'services/api/schema';

export const workflowsApi = api.injectEndpoints({
  endpoints: (build) => ({
    getWorkflow: build.query<WorkflowV2 | undefined, string>({
      query: (workflow_id) => `workflows/i/${workflow_id}`,
      providesTags: (result, error, workflow_id) => [
        { type: 'Workflow', id: workflow_id },
      ],
      transformResponse: (
        response: paths['/api/v1/workflows/i/{workflow_id}']['get']['responses']['200']['content']['application/json']
      ) => {
        if (response) {
          const result = zWorkflowV2.safeParse(response);
          if (result.success) {
            return result.data;
          } else {
            logger('images').warn('Problem parsing workflow');
          }
        }
        return;
      },
    }),
  }),
});

export const { useGetWorkflowQuery } = workflowsApi;
