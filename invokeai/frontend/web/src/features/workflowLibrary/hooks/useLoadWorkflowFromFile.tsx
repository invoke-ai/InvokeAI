import { useAppDispatch } from 'app/store/storeHooks';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { useValidateAndLoadWorkflow } from 'features/workflowLibrary/hooks/useValidateAndLoadWorkflow';
import { workflowLoadedFromFile } from 'features/workflowLibrary/store/actions';
import { useCallback } from 'react';

/**
 * Loads a workflow from a file.
 *
 * You probably should instead use `useLoadWorkflowWithDialog`, which opens a dialog to prevent loss of unsaved changes
 * and handles the loading process.
 */
export const useLoadWorkflowFromFile = () => {
  const dispatch = useAppDispatch();
  const validatedAndLoadWorkflow = useValidateAndLoadWorkflow();
  const loadWorkflowFromFile = useCallback(
    (
      file: File,
      options: {
        onSuccess?: (workflow: WorkflowV3) => void;
        onError?: () => void;
        onCompleted?: () => void;
      } = {}
    ) => {
      return new Promise<WorkflowV3 | void>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = async () => {
          const rawJSON = reader.result;
          const { onSuccess, onError, onCompleted } = options;
          try {
            const unvalidatedWorkflow = JSON.parse(rawJSON as string);
            const validatedWorkflow = await validatedAndLoadWorkflow(unvalidatedWorkflow, 'file');

            if (!validatedWorkflow) {
              reader.abort();
              onError?.();
              return;
            }
            dispatch(workflowLoadedFromFile());
            onSuccess?.(validatedWorkflow);
            resolve(validatedWorkflow);
          } catch (e) {
            // Log file parsing/loading errors
            if (e instanceof SyntaxError) {
              // eslint-disable-next-line no-console
              console.error('Workflow file JSON parsing failed:', {
                fileName: file.name,
                error: e.message,
              });
            } else {
              // eslint-disable-next-line no-console
              console.error('Workflow file loading failed:', {
                fileName: file.name,
                error: e,
              });
            }
            onError?.();
            reject();
          } finally {
            onCompleted?.();
          }
        };

        reader.readAsText(file);
      });
    },
    [validatedAndLoadWorkflow, dispatch]
  );

  return loadWorkflowFromFile;
};
