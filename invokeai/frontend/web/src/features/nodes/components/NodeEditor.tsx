import 'reactflow/dist/style.css';
import { Box } from '@chakra-ui/react';
import { ReactFlowProvider } from 'reactflow';

import { Flow } from './Flow';
import { memo } from 'react';

const NodeEditor = () => {
  return (
    <Box
      sx={{
        position: 'relative',
        width: 'full',
        height: { base: '100vh', xl: 'full' },
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

export default memo(NodeEditor);
