import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { chakra } from '@invoke-ai/ui-library';
import { memo } from 'react';
import type { PanelResizeHandleProps } from 'react-resizable-panels';
import { PanelResizeHandle } from 'react-resizable-panels';

const ChakraPanelResizeHandle = chakra(PanelResizeHandle);

const ResizeHandle = (props: Omit<PanelResizeHandleProps, 'style'>) => {
  return <ChakraPanelResizeHandle {...props} sx={sx} />;
};

export default memo(ResizeHandle);

const sx: SystemStyleObject = {
  '&[data-resize-handle-state="hover"]': {
    _before: {
      background: 'base.600 !important',
    },
  },
  '&[data-resize-handle-state="drag"]': {
    _before: {
      background: 'base.500 !important',
    },
  },
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
  '&[data-panel-group-direction="vertical"]': {
    h: 4,
    w: 'full',
    position: 'relative',
    _before: {
      transitionProperty: 'background',
      transitionDuration: 'normal',
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
