import { Box } from '@chakra-ui/react';
import ResizeHandle from 'features/ui/components/tabs/ResizeHandle';
import { memo, useState } from 'react';
import { Panel, PanelGroup } from 'react-resizable-panels';
import 'reactflow/dist/style.css';
import { Flow } from './Flow';
import NodeEditorPanelGroup from './panel/NodeEditorPanelGroup';

const NodeEditor = () => {
  const [isPanelCollapsed, setIsPanelCollapsed] = useState(false);
  return (
    <PanelGroup
      id="node-editor"
      autoSaveId="node-editor"
      direction="horizontal"
      style={{ height: '100%', width: '100%' }}
    >
      <Panel
        id="node-editor-panel-group"
        collapsible
        onCollapse={setIsPanelCollapsed}
        minSize={25}
      >
        <NodeEditorPanelGroup />
      </Panel>
      <ResizeHandle
        collapsedDirection={isPanelCollapsed ? 'left' : undefined}
      />
      <Panel id="node-editor-content">
        <Box
          layerStyle={'first'}
          sx={{
            position: 'relative',
            width: 'full',
            height: 'full',
            borderRadius: 'base',
          }}
        >
          <Flow />
        </Box>
      </Panel>
    </PanelGroup>
  );
};

export default memo(NodeEditor);
