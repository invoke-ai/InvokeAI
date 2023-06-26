import 'reactflow/dist/style.css';
import { Box } from '@chakra-ui/react';
import { ReactFlowProvider } from 'reactflow';

import { Flow } from './Flow';
import { memo } from 'react';

const NodeEditor = () => {
  return (
    <Box
      layerStyle={'first'}
      sx={{
        position: 'relative',
        width: 'full',
        height: 'full',
        borderRadius: 'base',
      }}
    >
      <ReactFlowProvider>
        <Flow />
      </ReactFlowProvider>
    </Box>
  );
};

export default memo(NodeEditor);
