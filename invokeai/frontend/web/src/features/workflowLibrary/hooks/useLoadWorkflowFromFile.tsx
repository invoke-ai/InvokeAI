import { useAppDispatch } from 'app/store/storeHooks';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { useValidateAndLoadWorkflow } from 'features/workflowLibrary/hooks/useValidateAndLoadWorkflow';
import { workflowLoadedFromFile } from 'features/workflowLibrary/store/actions';
import { useCallback } from 'react';

export const useLoadWorkflowFromFile = () => {
  const dispatch = useAppDispatch();
  const validatedAndLoadWorkflow = useValidateAndLoadWorkflow();
  const loadWorkflowFromFile = useCallback(
    (
      file: File,
      options: {
        onSuccess?: (workflow: WorkflowV3) => void;
        onError?: () => void;
      } = {}
    ) => {
      const reader = new FileReader();
      reader.onload = async () => {
        const rawJSON = reader.result;
        const { onSuccess, onError } = options;
        try {
          const unvalidatedWorkflow = JSON.parse(rawJSON as string);
          const validatedWorkflow = await validatedAndLoadWorkflow(unvalidatedWorkflow);

          if (!validatedWorkflow) {
            reader.abort();
            onError?.();
            return;
          }
          dispatch(workflowLoadedFromFile());
          onSuccess?.(validatedWorkflow);
        } catch {
          // This is catching the error from the parsing the JSON file
          onError?.();
        }
      };

      reader.readAsText(file);
    },
    [validatedAndLoadWorkflow, dispatch]
  );

  return loadWorkflowFromFile;
};
