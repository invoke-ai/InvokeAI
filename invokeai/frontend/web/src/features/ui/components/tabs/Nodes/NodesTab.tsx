import NodeEditor from 'features/nodes/components/NodeEditor';
import { memo } from 'react';
import { ReactFlowProvider } from 'reactflow';

const NodesTab = () => {
  return (
    <ReactFlowProvider>
      <NodeEditor />
    </ReactFlowProvider>
  );
};

export default memo(NodesTab);
