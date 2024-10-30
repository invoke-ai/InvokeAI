import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable, dropTargetForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { attachClosestEdge, extractClosestEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/closest-edge';
import type { DndListState } from 'features/dnd/dnd';
import { buildDndSourceApi, idle } from 'features/dnd/dnd';
import type { FieldIdentifier } from 'features/nodes/types/field';
import type { RefObject } from 'react';
import { useEffect, useState } from 'react';

/**
 * Dnd source API for a single workflow field.
 */
export const singleWorkflowField = buildDndSourceApi<{ fieldIdentifier: FieldIdentifier }>('SingleWorkflowField');

export const useLinearViewFieldDnd = (ref: RefObject<HTMLElement>, fieldIdentifier: FieldIdentifier) => {
  const [dndState, setDndState] = useState<DndListState>(idle);

  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }
    return combine(
      draggable({
        element,
        getInitialData() {
          return singleWorkflowField.getData({ fieldIdentifier });
        },
        onDragStart() {
          setDndState({ type: 'is-dragging' });
        },
        onDrop() {
          setDndState(idle);
        },
      }),
      dropTargetForElements({
        element,
        canDrop({ source }) {
          if (!singleWorkflowField.typeGuard(source.data)) {
            return false;
          }
          return true;
        },
        getData({ input }) {
          const data = singleWorkflowField.getData({ fieldIdentifier });
          return attachClosestEdge(data, {
            element,
            input,
            allowedEdges: ['top', 'bottom'],
          });
        },
        getIsSticky() {
          return true;
        },
        onDragEnter({ self }) {
          const closestEdge = extractClosestEdge(self.data);
          setDndState({ type: 'is-dragging-over', closestEdge });
        },
        onDrag({ self }) {
          const closestEdge = extractClosestEdge(self.data);

          // Only need to update react state if nothing has changed.
          // Prevents re-rendering.
          setDndState((current) => {
            if (current.type === 'is-dragging-over' && current.closestEdge === closestEdge) {
              return current;
            }
            return { type: 'is-dragging-over', closestEdge };
          });
        },
        onDragLeave() {
          setDndState(idle);
        },
        onDrop() {
          setDndState(idle);
        },
      })
    );
  }, [fieldIdentifier, ref]);

  return dndState;
};
