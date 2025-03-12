import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { graphToWorkflow } from 'features/nodes/util/workflow/graphToWorkflow';
import { toast } from 'features/toast/toast';
import { useValidateAndLoadWorkflow } from 'features/workflowLibrary/hooks/useValidateAndLoadWorkflow';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useLazyGetImageWorkflowQuery } from 'services/api/endpoints/images';
import type { NonNullableGraph } from 'services/api/types';
import { assert } from 'tsafe';

export const useLoadWorkflowFromImage = () => {
  const { t } = useTranslation();
  const [getWorkflowAndGraphFromImage, result] = useLazyGetImageWorkflowQuery();
  const validateAndLoadWorkflow = useValidateAndLoadWorkflow();
  const getAndLoadEmbeddedWorkflow = useCallback(
    async (
      imageName: string,
      options: {
        onSuccess?: (workflow: WorkflowV3) => void;
        onError?: () => void;
      } = {}
    ) => {
      const { onSuccess, onError } = options;
      try {
        const { workflow, graph } = await getWorkflowAndGraphFromImage(imageName).unwrap();

        // Images may have a workflow and/or a graph. We can load either into the workflow editor, but we prefer the
        // workflow.
        const unvalidatedWorkflow = workflow
          ? JSON.parse(workflow)
          : graph
            ? graphToWorkflow(JSON.parse(graph) as NonNullableGraph, true)
            : null;

        assert(unvalidatedWorkflow !== null, 'No workflow or graph provided');

        const validatedWorkflow = await validateAndLoadWorkflow(unvalidatedWorkflow);

        if (!validatedWorkflow) {
          onError?.();
          return;
        }

        onSuccess?.(validatedWorkflow);
      } catch {
        // This is catching:
        // - the error from the getWorkflowAndGraphFromImage query
        // - the error from parsing the workflow or graph
        toast({
          id: 'PROBLEM_RETRIEVING_WORKFLOW',
          title: t('toast.problemRetrievingWorkflow'),
          status: 'error',
        });
        onError?.();
        return;
      }
    },
    [getWorkflowAndGraphFromImage, validateAndLoadWorkflow, t]
  );

  return [getAndLoadEmbeddedWorkflow, result] as const;
};
