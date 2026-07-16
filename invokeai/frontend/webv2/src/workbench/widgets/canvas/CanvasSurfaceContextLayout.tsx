import type { BoxProps } from '@chakra-ui/react';
import type { MouseEventHandler, ReactNode } from 'react';

import { Box } from '@chakra-ui/react';

const CANVAS_SURFACE_LAYER_LAYOUT = {
  inset: '0',
  position: 'absolute',
} satisfies BoxProps;

export const CanvasSurfaceContextLayout = ({
  children,
  onContextMenu,
  surface,
}: {
  children: ReactNode;
  onContextMenu: MouseEventHandler<HTMLDivElement>;
  surface: ReactNode;
}) => (
  <>
    <Box {...CANVAS_SURFACE_LAYER_LAYOUT} data-canvas-context-menu-owner="" onContextMenu={onContextMenu}>
      {surface}
    </Box>
    {children}
  </>
);
