import { useBuildWorkflowFast } from 'features/nodes/util/workflow/buildWorkflow';
import { saveWorkflowAs } from 'features/workflowLibrary/components/SaveWorkflowAsDialog';
import { isLibraryWorkflow, useSaveLibraryWorkflow } from 'features/workflowLibrary/hooks/useSaveLibraryWorkflow';
import { useCallback } from 'react';

/**
 * Returns a function that saves the current workflow if it's a library workflow, or opens the save dialog.
 */
export const useSaveOrSaveAsWorkflow = () => {
  const buildWorkflow = useBuildWorkflowFast();
  const { saveWorkflow } = useSaveLibraryWorkflow();

  const saveOrSaveAsWorkflow = useCallback(() => {
    const workflow = buildWorkflow();

    if (isLibraryWorkflow(workflow)) {
      saveWorkflow(workflow);
    } else {
      saveWorkflowAs(workflow);
    }
  }, [buildWorkflow, saveWorkflow]);

  return saveOrSaveAsWorkflow;
};
