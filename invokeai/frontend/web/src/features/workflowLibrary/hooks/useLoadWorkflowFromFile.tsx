import { useAppDispatch } from 'app/store/storeHooks';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { useLoadWorkflow } from 'features/workflowLibrary/hooks/useLoadWorkflow';
import { workflowLoadedFromFile } from 'features/workflowLibrary/store/actions';
import type { RefObject } from 'react';
import { useCallback } from 'react';
import { assert } from 'tsafe';

type useLoadWorkflowFromFileOptions = {
  resetRef: RefObject<() => void>;
  onSuccess?: (workflow: WorkflowV3) => void;
};

type UseLoadWorkflowFromFile = (options: useLoadWorkflowFromFileOptions) => (file: File | null) => void;

export const useLoadWorkflowFromFile: UseLoadWorkflowFromFile = ({ resetRef, onSuccess }) => {
  const dispatch = useAppDispatch();
  const loadWorkflow = useLoadWorkflow();
  const loadWorkflowFromFile = useCallback(
    (file: File | null) => {
      if (!file) {
        return;
      }
      const reader = new FileReader();
      reader.onload = async () => {
        const rawJSON = reader.result;

        try {
          const workflow = await loadWorkflow({ workflow: String(rawJSON), graph: null });
          assert(workflow !== null);
          dispatch(workflowLoadedFromFile());
          onSuccess && onSuccess(workflow);
        } catch (e) {
          reader.abort();
        }
      };

      reader.readAsText(file);

      // Reset the file picker internal state so that the same file can be loaded again
      resetRef.current?.();
    },
    [resetRef, loadWorkflow, dispatch, onSuccess]
  );

  return loadWorkflowFromFile;
};
