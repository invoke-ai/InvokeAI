import type { Edge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/closest-edge';

/**
 * States for a dnd target.
 * - `idle`: No drag is occurring, or the drag is not valid for the current drop target.
 * - `potential`: A drag is occurring, and the drag is valid for the current drop target, but the drag is not over the
 *  drop target.
 * - `over`: A drag is occurring, and the drag is valid for the current drop target, and the drag is over the drop target.
 */
export type DndTargetState = 'idle' | 'potential' | 'over';

/**
 * States for a dnd list.
 */
export type DndListTargetState =
  | {
      type: 'idle';
    }
  | {
      type: 'preview';
      container: HTMLElement;
    }
  | {
      type: 'is-dragging';
    }
  | {
      type: 'is-dragging-over';
      closestEdge: Edge | null;
    };
export const idle: DndListTargetState = { type: 'idle' };
