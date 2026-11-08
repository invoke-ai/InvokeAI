import { useAppDispatch } from 'app/store/storeHooks';
import { useBuildWorkflowFast } from 'features/nodes/util/workflow/buildWorkflow';
import { workflowDownloaded } from 'features/workflowLibrary/store/actions';
import { useCallback } from 'react';

export const useDownloadCurrentlyLoadedWorkflow = () => {
  const dispatch = useAppDispatch();
  const buildWorkflow = useBuildWorkflowFast();

  const downloadWorkflow = useCallback(() => {
    const workflow = buildWorkflow();

    const blob = new Blob([JSON.stringify(workflow, null, 2)]);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${workflow.name || 'My Workflow'}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    dispatch(workflowDownloaded());
  }, [buildWorkflow, dispatch]);

  return downloadWorkflow;
};
