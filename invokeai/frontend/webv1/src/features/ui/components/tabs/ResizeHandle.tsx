import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { chakra, Flex, Icon } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { PiDotsSix, PiDotsSixVertical } from 'react-icons/pi';
import type { PanelResizeHandleProps } from 'react-resizable-panels';
import { PanelResizeHandle } from 'react-resizable-panels';

const ChakraPanelResizeHandle = chakra(PanelResizeHandle);

const commonSx: SystemStyleObject = {
  '&[data-resize-handle-state="hover"]': {
    _before: {
      background: 'base.600 !important',
    },
    '.resize-handle-dots': {
      color: 'base.400',
    },
  },
  '&[data-resize-handle-state="drag"]': {
    _before: {
      background: 'base.500 !important',
    },
    '.resize-handle-dots': {
      color: 'base.300',
    },
  },
  '.resize-handle-dots': {
    pointerEvents: 'none',
    position: 'absolute',
    left: '50%',
    top: '50%',
    transform: 'translateX(-50%) translateY(-50%) scale(0.75)',
    background: 'base.900',
    color: 'base.500',
  },
};

const horizontalSx: SystemStyleObject = {
  ...commonSx,
  '&[data-panel-group-direction="vertical"]': {
    h: 4,
    w: 'full',
    position: 'relative',
    _before: {
      transitionProperty: 'background',
      transitionDuration: 'fast',
      content: '""',
      w: 'full',
      h: '2px',
      background: 'base.800',
      position: 'absolute',
      top: '50%',
      left: 0,
      transform: 'translateY(-50%)',
    },
  },
};

export const HorizontalResizeHandle = memo((props: Omit<PanelResizeHandleProps, 'style'>) => {
  return (
    <ChakraPanelResizeHandle {...props} sx={horizontalSx}>
      <Flex className="resize-handle-dots" clipPath="inset(0px 2px 2px 0px)" px={1}>
        <Icon as={PiDotsSix} me={-0.5} />
        <Icon as={PiDotsSix} ms={-0.5} />
      </Flex>
    </ChakraPanelResizeHandle>
  );
});
HorizontalResizeHandle.displayName = 'HorizontalResizeHandle';

const verticalSx: SystemStyleObject = {
  ...commonSx,
  '&[data-panel-group-direction="horizontal"]': {
    w: 4,
    h: 'full',
    position: 'relative',
    _before: {
      transitionProperty: 'background',
      transitionDuration: 'normal',
      content: '""',
      w: '2px',
      h: 'full',
      background: 'base.800',
      position: 'absolute',
      left: '50%',
      top: 0,
      transform: 'translateX(-50%)',
    },
  },
};

export const VerticalResizeHandle = memo((props: Omit<PanelResizeHandleProps, 'style'>) => {
  return (
    <ChakraPanelResizeHandle {...props} sx={verticalSx}>
      <Flex flexDir="column" className="resize-handle-dots" clipPath="inset(2px 0px 0px 2px)" py={1}>
        <Icon as={PiDotsSixVertical} mb={-0.5} />
        <Icon as={PiDotsSixVertical} mt={-0.5} />
      </Flex>
    </ChakraPanelResizeHandle>
  );
});
VerticalResizeHandle.displayName = 'VerticalResizeHandle';
