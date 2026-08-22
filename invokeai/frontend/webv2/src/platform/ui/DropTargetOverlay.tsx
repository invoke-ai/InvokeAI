import { Text } from '@chakra-ui/react';

import { DropZone } from './DropZone';

/**
 * The drag-in-flight affordance every image/video drop target shares: renders
 * nothing until a compatible drag is active anywhere, then floats the
 * `DropZone` overlay treatment (dashed border over a surface tint, flipping
 * to accent while hovered) above the target's own content with a centered
 * call-to-action — the "potential target" / "over" pair of states.
 *
 * `pointerEvents` stays off — dnd-kit hit-tests the droppable's rect, not
 * this overlay — and the absolute inset needs `position: relative` (or any
 * containing block) on the target element.
 */
export const DropTargetOverlay = ({
  isActive,
  isOver,
  label,
}: {
  /** A compatible drag is in flight somewhere; the overlay exists only then. */
  isActive: boolean;
  /** That drag is hovering this target right now. */
  isOver?: boolean;
  /** Centered call-to-action; omit for targets too small to carry text. */
  label?: string;
}) =>
  isActive ? (
    <DropZone
      alignItems="center"
      display="flex"
      inset="0"
      isOver={isOver}
      justifyContent="center"
      pointerEvents="none"
      position="absolute"
      variant="overlay"
      zIndex="2"
    >
      {label ? (
        <Text color="fg" fontSize="sm" fontWeight="700" px="2" textAlign="center" textWrap="pretty">
          {label}
        </Text>
      ) : null}
    </DropZone>
  ) : null;
