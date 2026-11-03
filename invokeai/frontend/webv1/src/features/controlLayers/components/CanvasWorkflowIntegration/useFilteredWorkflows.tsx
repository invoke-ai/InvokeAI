import { logger } from 'app/logging/logger';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEffect, useMemo, useState } from 'react';
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
  const workflowIds = useMemo(() => workflows.map((workflow) => workflow.workflow_id), [workflows]);
  const [filteredWorkflowIds, setFilteredWorkflowIds] = useState<Set<string>>(() => new Set());
  const [isFiltering, setIsFiltering] = useState(false);

  const filteredWorkflows = useMemo(
    () => workflows.filter((workflow) => filteredWorkflowIds.has(workflow.workflow_id)),
    [filteredWorkflowIds, workflows]
  );

  useEffect(() => {
    let isCancelled = false;

    const filterWorkflows = async () => {
      if (workflows.length === 0) {
        setFilteredWorkflowIds(new Set());
        setIsFiltering(false);
        return;
      }

      setIsFiltering(true);

      try {
        const filteredWorkflowIds = new Set<string>();

        await Promise.all(
          workflows.map(async (workflow) => {
            try {
              const result = await dispatch(
                workflowsApi.endpoints.getWorkflow.initiate(workflow.workflow_id, {
                  subscribe: false,
                  forceRefetch: false,
                })
              );
              const data = 'data' in result ? result.data : undefined;
              const hasImageField = workflowHasImageField(data);

              log.debug(
                { workflowId: workflow.workflow_id, name: workflow.name, hasImageField },
                'Checked workflow for ImageField'
              );

              if ('unsubscribe' in result && typeof result.unsubscribe === 'function') {
                result.unsubscribe();
              }

              if (hasImageField) {
                filteredWorkflowIds.add(workflow.workflow_id);
              }
            } catch (error) {
              log.error(
                {
                  error: error instanceof Error ? error.message : String(error),
                  workflowId: workflow.workflow_id,
                },
                'Error checking workflow for ImageField'
              );
            }
          })
        );

        log.debug({ totalWorkflows: workflows.length, filteredCount: filteredWorkflowIds.size }, 'Filtered workflows');

        if (!isCancelled) {
          setFilteredWorkflowIds(filteredWorkflowIds);
        }
      } catch (error) {
        log.error({ error: error instanceof Error ? error.message : String(error) }, 'Error filtering workflows');
        if (!isCancelled) {
          setFilteredWorkflowIds(new Set());
        }
      } finally {
        if (!isCancelled) {
          setIsFiltering(false);
        }
      }
    };

    void filterWorkflows();

    return () => {
      isCancelled = true;
    };
  }, [dispatch, workflowIds, workflows]);

  return {
    filteredWorkflows,
    isFiltering,
  };
}
