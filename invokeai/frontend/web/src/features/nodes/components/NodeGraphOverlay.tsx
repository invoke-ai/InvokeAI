import { Box } from '@chakra-ui/react';
import { RootState } from 'app/store';
import { useAppSelector } from 'app/storeHooks';
import { buildNodesGraph } from '../util/nodesGraphBuilder/buildNodesGraph';

export default function NodeGraphOverlay() {
  const state = useAppSelector((state: RootState) => state);
  const graph = buildNodesGraph(state);

  return (
    <Box
      as="pre"
      fontFamily="monospace"
      position="absolute"
      top={10}
      right={2}
      userSelect="none"
      opacity={0.7}
      background="base.800"
      p={2}
      maxHeight={500}
      overflowY="scroll"
      borderRadius="md"
    >
      {JSON.stringify(graph, null, 2)}
    </Box>
  );
}
