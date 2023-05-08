import { Box, Flex } from '@chakra-ui/react';
import { memo } from 'react';
import { PanelResizeHandle } from 'react-resizable-panels';

type ResizeHandleProps = {
  direction?: 'horizontal' | 'vertical';
};

const ResizeHandle = (props: ResizeHandleProps) => {
  const { direction = 'horizontal' } = props;

  if (direction === 'horizontal') {
    return (
      <PanelResizeHandle>
        <Flex
          sx={{
            w: 6,
            h: 'full',
            justifyContent: 'center',
            alignItems: 'center',
          }}
        >
          <Box sx={{ w: 0.5, h: 'calc(100% - 4px)', bg: 'base.850' }} />
        </Flex>
      </PanelResizeHandle>
    );
  }
  return (
    <PanelResizeHandle>
      <Flex
        sx={{
          w: 'full',
          h: 6,
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        <Box sx={{ w: 'calc(100% - 4px)', h: 0.5, bg: 'base.850' }} />
      </Flex>
    </PanelResizeHandle>
  );
};

export default memo(ResizeHandle);
