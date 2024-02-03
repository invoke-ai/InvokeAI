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
        <Box className="resize-handle-drag-handle" data-orientation={orientation} />
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
      w: 'calc(100% - 1rem)',
      h: '2px',
    },
    '&[data-orientation="vertical"]': {
      w: '2px',
      h: 'calc(100% - 1rem)',
    },
    borderRadius: 'base',
    transitionProperty: 'inherit',
    transitionDuration: 'inherit',
  },
  '.resize-handle-drag-handle': {
    pos: 'absolute',
    borderRadius: '1px',
    transitionProperty: 'inherit',
    transitionDuration: 'inherit',
    '&[data-orientation="horizontal"]': {
      w: '30px',
      h: '6px',
      insetInlineStart: '50%',
      transform: 'translate(-50%, 0)',
    },
    '&[data-orientation="vertical"]': {
      w: '6px',
      h: '30px',
      insetBlockStart: '50%',
      transform: 'translate(0, -50%)',
    },
  },
};
