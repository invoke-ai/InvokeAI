// Adapted from https://github.com/alexreardon/pdnd-react-tailwind/blob/main/src/drop-indicator.tsx
import type { Edge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/types';
import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box } from '@invoke-ai/ui-library';
import type { DndListTargetState } from 'features/dnd/types';

/**
 * Design decisions for the drop indicator's main line
 */
const line = {
  thickness: 2,
  backgroundColor: 'red',
  // backgroundColor: 'base.500',
};

type DropIndicatorProps = {
  /**
   * The `edge` to draw a drop indicator on.
   *
   * `edge` is required as for the best possible performance
   * outcome you should only render this component when it needs to do something
   *
   * @example {closestEdge && <DropIndicator edge={closestEdge} />}
   */
  edge: Edge;
  /**
   * `gap` allows you to position the drop indicator further away from the drop target.
   * `gap` should be the distance between your drop targets
   * a drop indicator will be rendered halfway between the drop targets
   * (the drop indicator will be offset by half of the `gap`)
   *
   * `gap` should be a valid CSS length.
   * @example "8px"
   * @example "var(--gap)"
   */
  gap?: string;
};

const lineStyles: SystemStyleObject = {
  display: 'block',
  position: 'absolute',
  zIndex: 1,
  borderRadius: 'full',
  // Blocking pointer events to prevent the line from triggering drag events
  // Dragging over the line should count as dragging over the element behind it
  pointerEvents: 'none',
  background: line.backgroundColor,
};

type Orientation = 'horizontal' | 'vertical';

const orientationStyles: Record<Orientation, SystemStyleObject> = {
  horizontal: {
    height: `${line.thickness}px`,
    left: 2,
    right: 2,
  },
  vertical: {
    width: `${line.thickness}px`,
    top: 2,
    bottom: 2,
  },
};

const edgeToOrientationMap: Record<Edge, Orientation> = {
  top: 'horizontal',
  bottom: 'horizontal',
  left: 'vertical',
  right: 'vertical',
};

const edgeStyles: Record<Edge, SystemStyleObject> = {
  top: {
    top: 'var(--local-line-offset)',
  },
  right: {
    right: 'var(--local-line-offset)',
  },
  bottom: {
    bottom: 'var(--local-line-offset)',
  },
  left: {
    left: 'var(--local-line-offset)',
  },
};

/**
 * __Drop indicator__
 *
 * A drop indicator is used to communicate the intended resting place of the draggable item. The orientation of the drop indicator should always match the direction of the content flow.
 */
function DndDropIndicatorInternal({ edge, gap = '0px' }: DropIndicatorProps) {
  /**
   * To clearly communicate the resting place of a draggable item during a drag operation,
   * the drop indicator should be positioned half way between draggable items.
   */
  const lineOffset = `calc(-0.5 * (${gap} + ${line.thickness}px))`;

  const orientation = edgeToOrientationMap[edge];

  return (
    <Box
      sx={{ ...lineStyles, ...orientationStyles[orientation], ...edgeStyles[edge], '--local-line-offset': lineOffset }}
    />
  );
}

export const DndListDropIndicator = ({ dndState, gap }: { dndState: DndListTargetState; gap?: string }) => {
  if (dndState.type !== 'is-dragging-over') {
    return null;
  }

  if (!dndState.closestEdge) {
    return null;
  }

  return (
    <DndDropIndicatorInternal
      edge={dndState.closestEdge}
      // This is the gap between items in the list, used to calculate the position of the drop indicator
      gap={gap || 'var(--invoke-space-2)'}
    />
  );
};
