import 'reactflow/dist/style.css';
import { Box } from '@chakra-ui/react';
import { ReactFlowProvider } from 'reactflow';

import { Flow } from './Flow';
import { useAppSelector } from 'app/storeHooks';
import { RootState } from 'app/store';
import { buildNodesGraph } from '../util/buildNodesGraph';

const NodeEditor = () => {
  const state = useAppSelector((state: RootState) => state);

  const graph = buildNodesGraph(state);

  return (
    <Box
      sx={{
        position: 'relative',
        width: 'full',
        height: 'full',
        borderRadius: 'md',
        bg: 'base.850',
      }}
    >
      <ReactFlowProvider>
        <Flow />
      </ReactFlowProvider>
      <Box
        as="pre"
        fontFamily="monospace"
        position="absolute"
        top={2}
        left={2}
        width="full"
        height="full"
        userSelect="none"
        pointerEvents="none"
        opacity={0.7}
      >
        <Box w="50%">{JSON.stringify(graph, null, 2)}</Box>
      </Box>
    </Box>
  );
};

export default NodeEditor;
