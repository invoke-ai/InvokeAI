import 'reactflow/dist/style.css';
import { Box } from '@chakra-ui/react';
import { ReactFlowProvider } from 'reactflow';

import { Flow } from './Flow';

const NodeEditor = () => {
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
    </Box>
  );
};

export default NodeEditor;
