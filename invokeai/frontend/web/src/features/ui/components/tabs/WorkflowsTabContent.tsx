import { ReactFlowProvider } from '@xyflow/react';
import { useAppSelector } from 'app/store/storeHooks';
import { ImageViewer } from 'features/gallery/components/ImageViewer/ImageViewer';
import NodeEditor from 'features/nodes/components/NodeEditor';
import { ViewContextProvider } from 'features/nodes/contexts/ViewContext';
import { selectWorkflowMode } from 'features/nodes/store/workflowSlice';
import { memo } from 'react';

export const WorkflowsMainPanel = memo(() => {
  const mode = useAppSelector(selectWorkflowMode);

  if (mode === 'edit') {
    return (
      <ViewContextProvider view="edit-mode-nodes">
        <ReactFlowProvider>
          <NodeEditor />
        </ReactFlowProvider>
      </ViewContextProvider>
    );
  }

  return <ImageViewer />;
});

WorkflowsMainPanel.displayName = 'WorkflowsMainPanel';
