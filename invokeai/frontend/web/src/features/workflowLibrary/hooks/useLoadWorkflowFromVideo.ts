import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { graphToWorkflow } from 'features/nodes/util/workflow/graphToWorkflow';
import { toast } from 'features/toast/toast';
import { useValidateAndLoadWorkflow } from 'features/workflowLibrary/hooks/useValidateAndLoadWorkflow';
import { t } from 'i18next';
import { useCallback } from 'react';
import { useLazyGetVideoWorkflowQuery } from 'services/api/endpoints/videos';
import type { NonNullableGraph } from 'services/api/types';
import { assert } from 'tsafe';

/**
 * Loads a workflow from a generated video. Mirrors `useLoadWorkflowFromImage`.
 *
 * You probably should instead use `useLoadWorkflowWithDialog`, which opens a dialog to prevent loss of unsaved changes
 * and handles the loading process.
 */
export const useLoadWorkflowFromVideo = () => {
  const [getWorkflowAndGraphFromVideo] = useLazyGetVideoWorkflowQuery();
  const validateAndLoadWorkflow = useValidateAndLoadWorkflow();
  const loadWorkflowFromVideo = useCallback(
    async (
      videoName: string,
      options: {
        onSuccess?: (workflow: WorkflowV3) => void;
        onError?: () => void;
        onCompleted?: () => void;
      } = {}
    ) => {
      const { onSuccess, onError, onCompleted } = options;
      try {
        const { workflow, graph } = await getWorkflowAndGraphFromVideo(videoName).unwrap();

        // Videos may have a workflow and/or a graph. We can load either into the workflow editor, but we prefer the
        // workflow.
        const unvalidatedWorkflow = workflow
          ? JSON.parse(workflow)
          : graph
            ? graphToWorkflow(JSON.parse(graph) as NonNullableGraph, true)
            : null;

        assert(unvalidatedWorkflow !== null, 'No workflow or graph provided');

        const validatedWorkflow = await validateAndLoadWorkflow(unvalidatedWorkflow, 'image');

        if (!validatedWorkflow) {
          onError?.();
          return;
        }

        onSuccess?.(validatedWorkflow);
      } catch {
        toast({
          id: 'PROBLEM_RETRIEVING_WORKFLOW',
          title: t('toast.problemRetrievingWorkflow'),
          status: 'error',
        });
        onError?.();
      } finally {
        onCompleted?.();
      }
    },
    [getWorkflowAndGraphFromVideo, validateAndLoadWorkflow]
  );

  return loadWorkflowFromVideo;
};
