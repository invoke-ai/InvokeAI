import { Box, Flex } from '@chakra-ui/react';
import { memo } from 'react';
import { PanelResizeHandle } from 'react-resizable-panels';

const ResizeHandle = () => {
  return (
    <PanelResizeHandle>
      <Flex
        sx={{ w: 6, h: 'full', justifyContent: 'center', alignItems: 'center' }}
      >
        <Box sx={{ w: 0.5, h: 'calc(100% - 1rem)', py: 4, bg: 'base.800' }} />
      </Flex>
    </PanelResizeHandle>
  );
};

export default memo(ResizeHandle);
