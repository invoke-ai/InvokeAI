import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable, dropTargetForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { attachClosestEdge, extractClosestEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/closest-edge';
import { singleWorkflowFieldDndSource } from 'features/dnd/dnd';
import type { DndListTargetState } from 'features/dnd/types';
import { idle } from 'features/dnd/types';
import { firefoxDndFix } from 'features/dnd/util';
import type { FieldIdentifier } from 'features/nodes/types/field';
import type { RefObject } from 'react';
import { useEffect, useState } from 'react';

export const useLinearViewFieldDnd = (ref: RefObject<HTMLElement>, fieldIdentifier: FieldIdentifier) => {
  const [dndListState, setListDndState] = useState<DndListTargetState>(idle);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }
    return combine(
      firefoxDndFix(element),
      draggable({
        element,
        getInitialData() {
          return singleWorkflowFieldDndSource.getData({ fieldIdentifier });
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
          if (!singleWorkflowFieldDndSource.typeGuard(source.data)) {
            return false;
          }
          return true;
        },
        getData({ input }) {
          const data = singleWorkflowFieldDndSource.getData({ fieldIdentifier });
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
