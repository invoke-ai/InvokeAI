import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { useValidateAndLoadWorkflow } from 'features/workflowLibrary/hooks/useValidateAndLoadWorkflow';
import { useCallback } from 'react';

/**
 * Loads a workflow from an object.
 *
 * You probably should instead use `useLoadWorkflowWithDialog`, which opens a dialog to prevent loss of unsaved changes
 * and handles the loading process.
 */
export const useLoadWorkflowFromObject = () => {
  const validateAndLoadWorkflow = useValidateAndLoadWorkflow();
  const loadWorkflowFromObject = useCallback(
    async (
      unvalidatedWorkflow: unknown,
      options: {
        onSuccess?: (workflow: WorkflowV3) => void;
        onError?: () => void;
        onCompleted?: () => void;
      } = {}
    ) => {
      const { onSuccess, onError, onCompleted } = options;
      try {
        const validatedWorkflow = await validateAndLoadWorkflow(unvalidatedWorkflow);

        if (!validatedWorkflow) {
          onError?.();
          return;
        }
        onSuccess?.(validatedWorkflow);
      } finally {
        onCompleted?.();
      }
    },
    [validateAndLoadWorkflow]
  );

  return loadWorkflowFromObject;
};
