import { Box } from '@chakra-ui/react';
import { ReactFlowProvider } from 'reactflow';
import 'reactflow/dist/style.css';

import { memo } from 'react';
import { Flow } from './Flow';

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
