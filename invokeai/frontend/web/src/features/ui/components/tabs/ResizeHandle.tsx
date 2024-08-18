import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, chakra, Flex } from '@invoke-ai/ui-library';
import { memo } from 'react';
import type { PanelResizeHandleProps } from 'react-resizable-panels';
import { PanelResizeHandle } from 'react-resizable-panels';

type ResizeHandleProps = PanelResizeHandleProps & {
  orientation: 'horizontal' | 'vertical';
};

const ChakraPanelResizeHandle = chakra(PanelResizeHandle);

const ResizeHandle = (props: ResizeHandleProps) => {
  const { orientation, ...rest } = props;

  return (
    <ChakraPanelResizeHandle {...rest}>
      <Flex sx={sx} data-orientation={orientation}>
        <Box className="resize-handle-inner" data-orientation={orientation} />
      </Flex>
    </ChakraPanelResizeHandle>
  );
};

export default memo(ResizeHandle);

const sx: SystemStyleObject = {
  display: 'flex',
  pos: 'relative',
  '&[data-orientation="horizontal"]': {
    w: 'full',
    h: 5,
  },
  '&[data-orientation="vertical"]': { w: 5, h: 'full' },
  alignItems: 'center',
  justifyContent: 'center',
  div: {
    bg: 'base.800',
  },
  _hover: {
    div: { bg: 'base.700' },
  },
  _active: {
    div: { bg: 'base.600' },
  },
  transitionProperty: 'common',
  transitionDuration: 'normal',
  '.resize-handle-inner': {
    '&[data-orientation="horizontal"]': {
      w: '100%',
      h: '2px',
    },
    '&[data-orientation="vertical"]': {
      w: '2px',
      h: '100%',
    },
    borderRadius: 'base',
    transitionProperty: 'inherit',
    transitionDuration: 'inherit',
  },
};
