import { Box } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';
import { buildNodesGraph } from '../util/graphBuilders/buildNodesGraph';

const NodeGraphOverlay = () => {
  const state = useAppSelector((state: RootState) => state);
  const graph = buildNodesGraph(state);

  return (
    <Box
      as="pre"
      sx={{
        fontFamily: 'monospace',
        position: 'absolute',
        top: 2,
        right: 2,
        opacity: 0.7,
        p: 2,
        maxHeight: 500,
        maxWidth: 500,
        overflowY: 'scroll',
        borderRadius: 'base',
        bg: 'base.200',
        _dark: { bg: 'base.800' },
      }}
    >
      {JSON.stringify(graph, null, 2)}
    </Box>
  );
};

export default memo(NodeGraphOverlay);
