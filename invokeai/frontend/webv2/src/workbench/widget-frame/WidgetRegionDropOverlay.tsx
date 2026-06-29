import type { WidgetRegionDropState } from '@workbench/widgetDnd';

import { Box } from '@chakra-ui/react';

export const WidgetRegionDropOverlay = ({
  dropState,
  isOver,
}: {
  dropState: WidgetRegionDropState;
  isOver: boolean;
}) => (
  <Box
    bg={dropState.isAllowed ? (isOver ? 'accent.subtle' : 'transparent') : 'bg.muted'}
    borderColor={dropState.isAllowed ? 'accent.solid' : 'border.subtle'}
    borderStyle="dashed"
    borderWidth="2px"
    bottom="0"
    left="0"
    opacity={dropState.isAllowed ? 0.96 : 0.5}
    pointerEvents="none"
    position="absolute"
    right="0"
    rounded="sm"
    shadow={dropState.isAllowed && isOver ? '0 0 0 1px {colors.accent.solid}' : undefined}
    top="0"
    transition="background var(--wb-motion-duration-fast) ease, border-color var(--wb-motion-duration-fast) ease, opacity var(--wb-motion-duration-fast) ease, box-shadow var(--wb-motion-duration-fast) ease"
    zIndex="2"
  />
);
