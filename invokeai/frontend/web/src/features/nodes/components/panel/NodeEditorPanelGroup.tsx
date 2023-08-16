import InspectorPanel from 'features/nodes/components/panel/InspectorPanel';
import ResizeHandle from 'features/ui/components/tabs/ResizeHandle';
import { memo, useState } from 'react';
import { Panel, PanelGroup } from 'react-resizable-panels';
import 'reactflow/dist/style.css';
import WorkflowPanel from './WorkflowPanel';

const NodeEditorPanelGroup = () => {
  const [isTopPanelCollapsed, setIsTopPanelCollapsed] = useState(false);
  const [isBottomPanelCollapsed, setIsBottomPanelCollapsed] = useState(false);

  return (
    <PanelGroup
      id="node-editor-panel_group"
      autoSaveId="node-editor-panel_group"
      direction="vertical"
      style={{ height: '100%', width: '100%' }}
    >
      <Panel
        id="node-editor-panel_group_workflow"
        collapsible
        onCollapse={setIsTopPanelCollapsed}
        minSize={25}
      >
        <WorkflowPanel />
      </Panel>
      <ResizeHandle
        direction="vertical"
        collapsedDirection={
          isTopPanelCollapsed
            ? 'top'
            : isBottomPanelCollapsed
            ? 'bottom'
            : undefined
        }
      />
      <Panel
        id="node-editor-panel_group_inspector"
        collapsible
        onCollapse={setIsBottomPanelCollapsed}
        minSize={25}
      >
        <InspectorPanel />
      </Panel>
    </PanelGroup>
  );
};

export default memo(NodeEditorPanelGroup);
