import { useWorkflow } from 'features/nodes/hooks/useWorkflow';
import { useCallback } from 'react';

export const useDownloadWorkflow = () => {
  const workflow = useWorkflow();
  const downloadWorkflow = useCallback(() => {
    const blob = new Blob([JSON.stringify(workflow, null, 2)]);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${workflow.name || 'My Workflow'}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }, [workflow]);

  return downloadWorkflow;
};
