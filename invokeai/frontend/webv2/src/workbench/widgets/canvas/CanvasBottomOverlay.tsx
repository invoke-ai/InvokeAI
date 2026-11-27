import type { BoxProps, StackProps } from '@chakra-ui/react';

import { Box, Stack } from '@chakra-ui/react';
import { forwardRef } from 'react';

export const BOTTOM_OVERLAY_LAYOUT = {
  bottom: '2',
  left: '2',
  minH: '0',
  overflow: 'hidden',
  pointerEvents: 'none',
  position: 'absolute',
  right: '2',
  top: '2',
  zIndex: '3',
} satisfies BoxProps;

export const BOTTOM_OVERLAY_STACK_LAYOUT = {
  align: 'center',
  h: 'full',
  justifyContent: 'flex-end',
  minH: '0',
  overflow: 'hidden',
} satisfies StackProps;

/** Stretch staging to canvas width so thumbnails scroll while contained bars remain centered. */
export const BOTTOM_STAGING_SLOT_LAYOUT = {
  flexShrink: '0',
  minW: '0',
  w: 'full',
} satisfies BoxProps;

const Root = forwardRef<HTMLDivElement, BoxProps>(({ children, ...props }, ref) => (
  <Box ref={ref} {...BOTTOM_OVERLAY_LAYOUT} {...props}>
    <Stack {...BOTTOM_OVERLAY_STACK_LAYOUT} gap="2">
      {children}
    </Stack>
  </Box>
));
Root.displayName = 'CanvasBottomOverlay.Root';

const Staging = (props: BoxProps) => <Box {...BOTTOM_STAGING_SLOT_LAYOUT} {...props} />;

export const CanvasBottomOverlay = { Root, Staging };
