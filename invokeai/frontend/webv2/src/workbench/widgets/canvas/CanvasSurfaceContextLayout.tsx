import type { BoxProps } from '@chakra-ui/react';
import type { KeyboardEventHandler, MouseEventHandler, ReactNode } from 'react';

import { Box } from '@chakra-ui/react';

const CANVAS_SURFACE_LAYER_LAYOUT = {
  inset: '0',
  position: 'absolute',
} satisfies BoxProps;

export const CanvasSurfaceContextLayout = ({
  children,
  onContextMenu,
  onKeyDown,
  surface,
}: {
  children: ReactNode;
  onContextMenu: MouseEventHandler<HTMLDivElement>;
  onKeyDown?: KeyboardEventHandler<HTMLDivElement>;
  surface: ReactNode;
}) => (
  <>
    <Box
      {...CANVAS_SURFACE_LAYER_LAYOUT}
      data-canvas-context-menu-owner=""
      onContextMenu={onContextMenu}
      onKeyDown={onKeyDown}
    >
      {surface}
    </Box>
    {children}
  </>
);
