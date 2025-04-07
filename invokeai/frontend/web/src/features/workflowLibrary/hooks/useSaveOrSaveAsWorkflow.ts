import { useIsWorkflowPublished } from 'features/nodes/components/sidePanel/workflow/publish';
import { useBuildWorkflowFast } from 'features/nodes/util/workflow/buildWorkflow';
import { saveWorkflowAs } from 'features/workflowLibrary/components/SaveWorkflowAsDialog';
import { isLibraryWorkflow, useSaveLibraryWorkflow } from 'features/workflowLibrary/hooks/useSaveLibraryWorkflow';
import { useCallback } from 'react';

/**
 * Returns a function that saves the current workflow if it's a library workflow, or opens the save as dialog.
 *
 * Published workflows are always saved as a new workflow.
 */
export const useSaveOrSaveAsWorkflow = () => {
  const buildWorkflow = useBuildWorkflowFast();
  const isPublished = useIsWorkflowPublished();
  const { saveWorkflow } = useSaveLibraryWorkflow();

  const saveOrSaveAsWorkflow = useCallback(() => {
    const workflow = buildWorkflow();

    if (isLibraryWorkflow(workflow) && !isPublished) {
      saveWorkflow(workflow);
    } else {
      saveWorkflowAs(workflow);
    }
  }, [buildWorkflow, isPublished, saveWorkflow]);

  return saveOrSaveAsWorkflow;
};
