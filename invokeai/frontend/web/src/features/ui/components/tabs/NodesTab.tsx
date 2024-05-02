import { Box, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import CurrentImageDisplay from 'features/gallery/components/CurrentImage/CurrentImageDisplay';
import { ImageViewer } from 'features/gallery/components/ImageViewer';
import NodeEditor from 'features/nodes/components/NodeEditor';
import { memo } from 'react';
import { ReactFlowProvider } from 'reactflow';

const NodesTab = () => {
  const mode = useAppSelector((s) => s.workflow.mode);

  if (mode === 'edit') {
    return (
      <Box layerStyle="first" position="relative" w="full" h="full" p={2} borderRadius="base">
        <ReactFlowProvider>
          <NodeEditor />
        </ReactFlowProvider>
        <ImageViewer />
      </Box>
    );
  }

  return (
    <Box layerStyle="first" position="relative" w="full" h="full" p={2} borderRadius="base">
      <Flex w="full" h="full">
        <CurrentImageDisplay />
      </Flex>
    </Box>
  );
};

export default memo(NodesTab);
