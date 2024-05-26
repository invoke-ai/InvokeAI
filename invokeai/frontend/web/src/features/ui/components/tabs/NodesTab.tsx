import { Box } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { ImageViewerWorkflows } from 'features/gallery/components/ImageViewer/ImageViewerWorkflows';
import NodeEditor from 'features/nodes/components/NodeEditor';
import { memo } from 'react';
import { ReactFlowProvider } from 'reactflow';

const NodesTab = () => {
  const mode = useAppSelector((s) => s.workflow.mode);
  if (mode === 'view') {
    return (
      <Box layerStyle="first" position="relative" w="full" h="full" p={2} borderRadius="base">
        <ImageViewerWorkflows />
      </Box>
    );
  }

  return (
    <Box layerStyle="first" position="relative" w="full" h="full" p={2} borderRadius="base">
      <ReactFlowProvider>
        <NodeEditor />
      </ReactFlowProvider>
    </Box>
  );
};

export default memo(NodesTab);
