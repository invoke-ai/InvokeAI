import type { BoxProps } from '@chakra-ui/react';
import type { Ref } from 'react';

import { Box } from '@chakra-ui/react';

const DROP_ZONE_TRANSITION =
  'background var(--wb-motion-duration-fast) ease, border-color var(--wb-motion-duration-fast) ease, opacity var(--wb-motion-duration-fast) ease, box-shadow var(--wb-motion-duration-fast) ease';

export interface DropZoneProps extends BoxProps {
  /** A compatible drag is hovering the zone. */
  isOver?: boolean;
  /** Forwarded to the underlying element (e.g. dnd-kit's `setNodeRef`). */
  ref?: Ref<HTMLDivElement>;
  /**
   * `inline` — a persistent, usually clickable upload box inside a form.
   * `overlay` — a drag-time overlay floated above existing content; heavier
   * border and a surface tint so it reads against arbitrary backdrops.
   */
  variant?: 'inline' | 'overlay';
}

/**
 * The workbench drop-target look: dashed `border.emphasized` at rest,
 * `accent.solid` border over an `accent.muted` tint while a compatible drag
 * hovers. Every drop zone and upload area composes this so drag affordances
 * stay identical across the app; callers add their own icon/hint content,
 * interaction handlers, and layout props.
 */
export const DropZone = ({ children, isOver, variant = 'inline', ...boxProps }: DropZoneProps) => (
  <Box
    bg={isOver ? 'accent.muted' : variant === 'overlay' ? 'bg.muted/80' : undefined}
    borderColor={isOver ? 'accent.solid' : 'border.emphasized'}
    borderStyle="dashed"
    borderWidth={variant === 'overlay' ? '2px' : '1px'}
    boxShadow={isOver && variant === 'overlay' ? '0 0 0 1px {colors.accent.solid}' : undefined}
    color={isOver ? 'fg' : 'fg.muted'}
    rounded="md"
    transition={DROP_ZONE_TRANSITION}
    {...boxProps}
  >
    {children}
  </Box>
);
