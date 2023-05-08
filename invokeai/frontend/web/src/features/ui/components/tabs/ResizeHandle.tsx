import { Box, Flex } from '@chakra-ui/react';
import { memo } from 'react';
import { PanelResizeHandle } from 'react-resizable-panels';

const ResizeHandle = () => {
  return (
    <PanelResizeHandle>
      <Flex
        sx={{ w: 6, h: 'full', justifyContent: 'center', alignItems: 'center' }}
      >
        <Box sx={{ w: 0.5, h: 'calc(100% - 4px)', bg: 'base.850' }} />
      </Flex>
    </PanelResizeHandle>
  );
};

export default memo(ResizeHandle);
