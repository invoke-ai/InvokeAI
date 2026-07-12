import type { BoxProps, FlexProps, StackProps } from '@chakra-ui/react';

import { Box, Flex, Stack } from '@chakra-ui/react';
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

export const BOTTOM_CONTROLS_SLOT_LAYOUT = {
  align: 'center',
  flex: '0 1 auto',
  justify: 'center',
  minH: '0',
  minW: '0',
  overflow: 'hidden',
  w: 'full',
} satisfies FlexProps;

const Root = forwardRef<HTMLDivElement, BoxProps>(({ children, ...props }, ref) => (
  <Box ref={ref} {...BOTTOM_OVERLAY_LAYOUT} {...props}>
    <Stack {...BOTTOM_OVERLAY_STACK_LAYOUT} gap="2">
      {children}
    </Stack>
  </Box>
));
Root.displayName = 'CanvasBottomOverlay.Root';

const Staging = (props: BoxProps) => <Box flexShrink="0" {...props} />;

const Controls = (props: FlexProps) => <Flex {...BOTTOM_CONTROLS_SLOT_LAYOUT} {...props} />;

export const CanvasBottomOverlay = { Controls, Root, Staging };
