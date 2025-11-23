import { logger } from 'app/logging/logger';
import { useAppDispatch } from 'app/store/storeHooks';
import { useCallback, useEffect, useState } from 'react';
import { workflowsApi } from 'services/api/endpoints/workflows';
import type { paths } from 'services/api/schema';

import { workflowHasImageField } from './workflowHasImageField';

const log = logger('canvas-workflow-integration');

type WorkflowListItem =
  paths['/api/v1/workflows/']['get']['responses']['200']['content']['application/json']['items'][number];

interface UseFilteredWorkflowsResult {
  filteredWorkflows: WorkflowListItem[];
  isFiltering: boolean;
}

/**
 * Hook that filters workflows to only include those with at least one ImageField
 * @param workflows The list of workflows to filter
 * @returns Filtered list of workflows that have ImageFields
 */
export function useFilteredWorkflows(workflows: WorkflowListItem[]): UseFilteredWorkflowsResult {
  const dispatch = useAppDispatch();
  const [filteredWorkflows, setFilteredWorkflows] = useState<WorkflowListItem[]>([]);
  const [isFiltering, setIsFiltering] = useState(false);

  const filterWorkflows = useCallback(async () => {
    if (workflows.length === 0) {
      setFilteredWorkflows([]);
      return;
    }

    setIsFiltering(true);

    try {
      // Load all workflows in parallel and check for ImageFields
      const workflowChecks = await Promise.all(
        workflows.map(async (workflow) => {
          try {
            // Fetch the full workflow data using dispatch
            const result = await dispatch(
              workflowsApi.endpoints.getWorkflow.initiate(workflow.workflow_id, {
                subscribe: false,
                forceRefetch: false,
              })
            );

            // Get the data from the result
            const data = 'data' in result ? result.data : undefined;

            const hasImageField = workflowHasImageField(data);

            log.debug(
              { workflowId: workflow.workflow_id, name: workflow.name, hasImageField },
              'Checked workflow for ImageField'
            );

            // Clean up the subscription
            if ('unsubscribe' in result && typeof result.unsubscribe === 'function') {
              result.unsubscribe();
            }

            return {
              workflow,
              hasImageField,
            };
          } catch (error) {
            log.error(
              {
                error: error instanceof Error ? error.message : String(error),
                workflowId: workflow.workflow_id,
              },
              'Error checking workflow for ImageField'
            );
            return {
              workflow,
              hasImageField: false,
            };
          }
        })
      );

      // Filter to only include workflows with ImageFields
      const filtered = workflowChecks.filter((check) => check.hasImageField).map((check) => check.workflow);

      log.debug({ totalWorkflows: workflows.length, filteredCount: filtered.length }, 'Filtered workflows');

      setFilteredWorkflows(filtered);
    } catch (error) {
      log.error({ error: error instanceof Error ? error.message : String(error) }, 'Error filtering workflows');
      setFilteredWorkflows([]);
    } finally {
      setIsFiltering(false);
    }
  }, [workflows, dispatch]);

  useEffect(() => {
    filterWorkflows();
  }, [filterWorkflows]);

  return {
    filteredWorkflows,
    isFiltering,
  };
}
