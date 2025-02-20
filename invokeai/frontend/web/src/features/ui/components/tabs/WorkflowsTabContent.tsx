import { ReactFlowProvider } from '@xyflow/react';
import { useAppSelector } from 'app/store/storeHooks';
import { ImageViewer } from 'features/gallery/components/ImageViewer/ImageViewer';
import NodeEditor from 'features/nodes/components/NodeEditor';
import { selectWorkflowMode } from 'features/nodes/store/workflowSlice';
import { memo } from 'react';

export const WorkflowsMainPanel = memo(() => {
  const mode = useAppSelector(selectWorkflowMode);

  if (mode === 'edit') {
    return (
      <ReactFlowProvider>
        <NodeEditor />
      </ReactFlowProvider>
    );
  }

  return <ImageViewer />;
});

WorkflowsMainPanel.displayName = 'WorkflowsMainPanel';
