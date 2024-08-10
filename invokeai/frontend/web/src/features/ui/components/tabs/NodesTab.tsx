import { Box } from '@invoke-ai/ui-library';
import { ReactFlowProvider } from '@xyflow/react';
import { useAppSelector } from 'app/store/storeHooks';
import { ImageComparisonDroppable } from 'features/gallery/components/ImageViewer/ImageComparisonDroppable';
import { ImageViewer } from 'features/gallery/components/ImageViewer/ImageViewer';
import NodeEditor from 'features/nodes/components/NodeEditor';
import { memo } from 'react';

const NodesTab = () => {
  const mode = useAppSelector((s) => s.workflow.mode);
  if (mode === 'view') {
    return (
      <Box layerStyle="first" position="relative" w="full" h="full" p={2} borderRadius="base">
        <ImageViewer />
        <ImageComparisonDroppable />
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
