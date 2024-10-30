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
  const [dndListState, setListDndState] = useState<DndListState>(idle);
  const [isDragging, setIsDragging] = useState(false);

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
          setListDndState({ type: 'is-dragging' });
          setIsDragging(true);
        },
        onDrop() {
          setListDndState(idle);
          setIsDragging(false);
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
          setListDndState({ type: 'is-dragging-over', closestEdge });
        },
        onDrag({ self }) {
          const closestEdge = extractClosestEdge(self.data);

          // Only need to update react state if nothing has changed.
          // Prevents re-rendering.
          setListDndState((current) => {
            if (current.type === 'is-dragging-over' && current.closestEdge === closestEdge) {
              return current;
            }
            return { type: 'is-dragging-over', closestEdge };
          });
        },
        onDragLeave() {
          setListDndState(idle);
        },
        onDrop() {
          setListDndState(idle);
        },
      })
    );
  }, [fieldIdentifier, ref]);

  return [dndListState, isDragging] as const;
};
