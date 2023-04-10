import 'reactflow/dist/style.css';
import { Box, HStack } from '@chakra-ui/react';
import { Flow } from './Flow';
import { useAppSelector } from 'app/storeHooks';
import { RootState } from 'app/store';
import { buildNodesGraph } from '../util/buildNodesGraph';
import { buildAdjacencyList } from '../util/isCyclic';

const NodeEditor = () => {
  const state = useAppSelector((state: RootState) => state);

  const graph = buildNodesGraph(state);

  const adjacencyList = buildAdjacencyList(state.nodes.edges);

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
      <Flow />
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
        <HStack alignItems={'flex-start'} justifyContent="space-between">
          <Box w="50%">{JSON.stringify(graph, null, 2)}</Box>
          <Box w="50%">{JSON.stringify(adjacencyList, null, 2)}</Box>
        </HStack>
      </Box>
    </Box>
  );
};

export default NodeEditor;
