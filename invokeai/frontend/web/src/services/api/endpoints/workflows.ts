import { logger } from 'app/logging/logger';
import { Workflow, zWorkflow } from 'features/nodes/types/types';
import { api } from '..';
import { paths } from '../schema';

export const workflowsApi = api.injectEndpoints({
  endpoints: (build) => ({
    getWorkflow: build.query<Workflow | undefined, string>({
      query: (workflow_id) => `workflows/i/${workflow_id}`,
      keepUnusedDataFor: 86400, // 24 hours
      providesTags: (result, error, workflow_id) => [
        { type: 'Workflow', id: workflow_id },
      ],
      transformResponse: (
        response: paths['/api/v1/workflows/i/{workflow_id}']['get']['responses']['200']['content']['application/json']
      ) => {
        if (response) {
          const result = zWorkflow.safeParse(response);
          if (result.success) {
            return result.data;
          } else {
            logger('images').warn('Problem parsing metadata');
          }
        }
        return;
      },
    }),
  }),
});

export const { useGetWorkflowQuery } = workflowsApi;
