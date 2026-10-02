import { Text } from '@chakra-ui/react';

import { DropZone } from './DropZone';

/** Requires a positioned parent. Keep pointer events off: dnd-kit hit-tests the droppable, not this overlay. */
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
