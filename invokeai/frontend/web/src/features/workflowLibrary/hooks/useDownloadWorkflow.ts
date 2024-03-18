import { useAppDispatch } from 'app/store/storeHooks';
import { $builtWorkflow } from 'features/nodes/hooks/useWorkflowWatcher';
import { workflowDownloaded } from 'features/workflowLibrary/store/actions';
import { useCallback } from 'react';

export const useDownloadWorkflow = () => {
  const dispatch = useAppDispatch();

  const downloadWorkflow = useCallback(() => {
    const workflow = $builtWorkflow.get();
    if (!workflow) {
      return;
    }
    const blob = new Blob([JSON.stringify(workflow, null, 2)]);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${workflow.name || 'My Workflow'}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    dispatch(workflowDownloaded());
  }, [dispatch]);

  return downloadWorkflow;
};
