import type { BoxProps } from '@chakra-ui/react';
import type { Ref } from 'react';

import { Box } from '@chakra-ui/react';
import { useMemo } from 'react';

const DROP_ZONE_TRANSITION =
  'background var(--wb-motion-duration-fast) ease, border-color var(--wb-motion-duration-fast) ease, opacity var(--wb-motion-duration-fast) ease, box-shadow var(--wb-motion-duration-fast) ease';

export interface DropZoneProps extends BoxProps {
  /** Native `disabled`, for zones rendered `as="button"`. */
  disabled?: boolean;
  /** The zone cannot accept input right now; the hover preview stands down. */
  isDisabled?: boolean;
  /** The last input was rejected; the error border survives pointer hover. */
  isInvalid?: boolean;
  /** A compatible drag is hovering the zone. */
  isOver?: boolean;
  /** Forwarded to the underlying element (e.g. dnd-kit's `setNodeRef`). */
  ref?: Ref<HTMLDivElement>;
  /** inline is a persistent upload surface; overlay floats above content during dragging. */
  variant?: 'inline' | 'overlay';
}

export const DropZone = ({
  children,
  isDisabled,
  isInvalid,
  isOver,
  variant = 'inline',
  _hover,
  ...boxProps
}: DropZoneProps) => {
  // Merge hover styles so they cannot override drag-over fills; overlays use isOver instead of pointer hover.
  const hoverProps = useMemo(
    () => ({
      ...(variant === 'inline' && !isDisabled ? { borderColor: 'accent.solid' } : null),
      ..._hover,
      ...(isInvalid && !isDisabled ? { bg: 'bg.error/60', borderColor: 'border.error' } : null),
      ...(isOver ? { bg: 'accent.solid/15', borderColor: 'accent.solid' } : null),
    }),
    [_hover, isDisabled, isInvalid, isOver, variant]
  );

  return (
    <Box
      bg={isOver ? 'accent.solid/15' : variant === 'overlay' ? 'bg.muted/60' : undefined}
      data-invalid={isInvalid || undefined}
      borderColor={isOver ? 'accent.solid' : isInvalid ? 'border.error' : 'border.emphasized'}
      borderStyle="dashed"
      borderWidth={variant === 'overlay' ? '2px' : '1px'}
      boxShadow={isOver && variant === 'overlay' ? '0 0 0 1px {colors.accent.solid}' : undefined}
      color={isOver ? 'fg' : 'fg.muted'}
      rounded="md"
      transition={DROP_ZONE_TRANSITION}
      _hover={hoverProps}
      {...boxProps}
    >
      {children}
    </Box>
  );
};
